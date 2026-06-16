package io.rolling.spark.ml.hs

import io.rolling.spark.ml.hs.HSModelStore.{load, store}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

/**
 * Anomaly Detection for time series with multiple seasonality and multiple metrics.
 *
 * Approach: additive time series regression with Fourier features (inspired by Prophet)
 *   - Piecewise linear trend with changepoints
 *   - Multiple seasonality via Fourier terms (hourly, weekly, monthly, yearly)
 *   - L2 linear regression (Spark MLlib) to estimate all coefficients
 *   - One regression per metric — features are shared, weights are independent
 *   - Anomaly = z_score of residuals beyond a configurable threshold
 *
 * Two clearly separated phases (no data leakage):
 *
 * PHASE 1 · Training (batch, scheduled weekly)
 *     - Reads N days of historical data
 *     - Computes minTs/rangeTs in HOURS on the training set and fixes them for scoring
 *     - Fits one LinearRegression per metric column inside a single Pipeline
 *     - Computes per-metric sigma from in-sample residuals
 *     - Saves PipelineModel + HSMeta to storage
 *
 * PHASE 2 · Scoring
 *     - Loads the saved PipelineModel (one artifact)
 *     - addFeatures → transform → classify for all metrics in one pass
 *     - Always uses minTs/rangeTs/cpPositions from training → consistent features
 *
 * Compatible with Spark 3.x, Scala 2.12/2.13
 *
 */
object HelterSkelter {

  /**
   * Instance of the logger class to correctly log at execution time
   */
  @transient protected lazy val logger: Logger = LogManager.getLogger(getClass)

  /** Fourier period constants in hours */
  val HOURS_PER_DAY: Double = 24.0
  val HOURS_PER_WEEK: Double = 168.0
  val HOURS_PER_MONTH: Double = 720.5
  val HOURS_PER_YEAR: Double = 8766.0

  /** Default sigma to use when training sigma cannot be computed (e.g. empty training set, perfect fit) to avoid division by zero during scoring. */
  val MIN_SIGMA = 1e-6

  /** Default z-score threshold for anomaly classification (2.5 = 2.5 standard deviations from the norm) */
  val DEFAULT_Z_THRESHOLD = 2.5

  // ─── PHASE 1: Batch training ──────────────────────────────────────────────────

  /**
   * Trains the HelterSkelter model on historical data.
   * One LinearRegression is fitted per metric column, all sharing the same Fourier/trend features.
   *
   * To be called periodically (e.g. Spark job scheduled weekly).
   *
   * Everything is computed in HOURS for consistency:
   *   - ts_hours = unix_timestamp / 3600
   *   - minTs, rangeTs, cpPositions → all in hours
   *   - Fourier periods (24h, 168h, ...) are divided directly by ts_hours
   *     without additional conversions
   *
   * @param historicDf DataFrame with historical data for training. Must have columns:
   *                   - timestamp (TimestampType, DateType, StringType with format yyyy-MM-dd HH:mm:ss,
   *                     or LongType/IntegerType interpreted as Unix epoch seconds)
   *                   - one or more numeric metric columns whose names are listed in `valueCols`
   * @param valueCols  set of column names in `historicDf` that contain the numeric metric values to model
   *                   (e.g. Set("bookings", "searches")). Each gets its own LinearRegression stage.
   * @param config     HSConfig with Fourier/changepoint hyper-parameters
   * @return HSModel containing the trained PipelineModel and HSMeta with training parameters and sigmas for scoring
   */
  def fit(historicDf: DataFrame, valueCols: Set[String], config: HSConfig = HSConfig()): HSModel = {

    logger.info("Starting training of HelterSkelter model")
    require(valueCols != null && valueCols.nonEmpty, "valueCols must not be empty")
    validateInput(historicDf, Set("timestamp") ++ valueCols, "train")

    // 1. Compute minTs and rangeTs in HOURS on the training set
    //    These two values are saved and reused unchanged during scoring.
    //    If they changed: trend of new records would no longer be comparable
    //    with trend seen during training → regression weights would not match.
    val tsStats = historicDf
      .withColumn("ts_hours", toTsHours(historicDf))
      .agg(min("ts_hours").as("min_ts_hours"), max("ts_hours").as("max_ts_hours"))
      .head()

    // agg(min, max) on an empty DataFrame returns one row with nulls — check explicitly
    require(!tsStats.isNullAt(0),
      "Training DataFrame is empty — cannot compute timestamp statistics")

    val minTs = tsStats.getAs[Double]("min_ts_hours") // first hour of training (e.g. 474480.0)
    val rangeTs = tsStats.getAs[Double]("max_ts_hours") - minTs // duration in hours (e.g. 2136.0 for 90 days)

    require(rangeTs > 0,
      s"Training data spans zero hours (all timestamps identical at $minTs). " +
        "Need at least two distinct timestamps to compute a trend.")

    // 2. Changepoint positions in HOURS, equidistant within the training set
    //    The regression will assign weight ≈ 0 to the unnecessary ones (L2 regularization)
    val cpPositions = if (config.nChangepoints > 0) {
      val cpStep = rangeTs / (config.nChangepoints + 1)
      (1 to config.nChangepoints).map(minTs + _ * cpStep)
    } else Seq.empty[Double]

    // 3. Add features to the training DataFrame and cache (reused by fit + sigma)
    val tmpMeta = HSMeta(
      minTs, rangeTs, null, cpPositions,
      fourierOrderHourly = config.fourierOrderHourly,
      fourierOrderWeekly = config.fourierOrderWeekly,
      fourierOrderMonthly = config.fourierOrderMonthly,
      fourierOrderYearly = config.fourierOrderYearly
    )

    val featured = addFeatures(historicDf, tmpMeta).cache()

    try {
      // 4. VectorAssembler: packs the feature columns into a single DenseVector per row.
      //    MLlib does not work on separate columns: each row must have a single vector.
      //    Example for a row:
      //      features = DenseVector(0.001, 0.0, ..., 0.866, 0.500, ..., 0.123)
      //                              ↑trend  ↑cp_1   ↑h_sin_1 ↑h_cos_1  ↑y_cos_8
      val assembler = new VectorAssembler()
        .setInputCols(featureColNames(config).toArray)
        .setOutputCol("features")

      val valueColList = valueCols.toList

      // 5. One L2 Linear Regression per metric column. Each finds weights w₀..wₙ that minimize:
      //      Σ (metric_i − w · features_i)²  +  λ · Σ wⱼ²
      //    Per metric, the weights independently learn:
      //      - w_trend    : base slope of the trend
      //      - w_cp_j     : slope change at changepoint j (δⱼ)
      //      - w_h_sin_k  : amplitude of the k-th hourly sinusoid
      //      - w_h_cos_k  : phase of the k-th hourly sinusoid
      //      - ...same for weekly, monthly, yearly
      //    regParam=0.01 (L2): penalizes large weights → prevents overfitting on Fourier terms
      //                        and reduces unnecessary changepoints to zero
      //    Each stage writes to its own prediction column (e.g. "bookings_prediction")
      //    so that multiple LR stages can coexist in one Pipeline without colliding.
      val lrStages = valueColList.map(valueCol => new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol(valueCol)
        .setPredictionCol(valueCol + "_prediction")
        .setMaxIter(200)
        .setRegParam(0.01)
        .setElasticNetParam(0.0) // 0.0 = pure L2 (Ridge), no L1
      )

      // 6. Single Pipeline: assembler + all LR stages
      val pipeline = new Pipeline().setStages((assembler :: lrStages).toArray)
      val pipelineModel = pipeline.fit(featured)

      // 7. Compute sigma per metric (reusing the cached featured DataFrame)
      //    For each metric: residual = observed − predicted
      //    sigma = stddev(residuals) = how much reality normally deviates from the model
      //    These values are saved and used in scoring as units of measure:
      //      z_score = new_residual / training_sigma
      //    If z_score > threshold → anomaly
      val transformed = pipelineModel.transform(featured).cache()
      val trainSigmas = try {
        val transformedWithResiduals = valueColList.foldLeft(transformed)((acc, valueCol) => acc.withColumn(valueCol + "_residual", col(valueCol) - col(valueCol + "_prediction")))
        val aggregatedCols = valueColList.map(valueCol => stddev(col(valueCol + "_residual")).as(valueCol + "_sigma"))
        val sigmaRowOpt = transformedWithResiduals.agg(aggregatedCols.head, aggregatedCols.tail: _*).head(1).headOption
        (sigmaRowOpt match {
          case None =>
            logger.warn(s"Could not compute training sigmas — no data after feature transformation. Falling back to $MIN_SIGMA for all metrics to avoid division by zero during scoring.")
            valueColList.map(_ -> MIN_SIGMA)
          case Some(row) =>
            valueColList.map(valueCol => {
              val sigma = row.getAs[Double](valueCol + "_sigma")
              val finalSigma =
                if (sigma > 0) {
                  logger.info(f"Training completed for value col $valueCol. In-sample sigma: $sigma%.2f")
                  sigma
                } else {
                  logger.warn(f"In-sample sigma for value col $valueCol is $sigma (perfect fit or insufficient data). Falling back to $MIN_SIGMA to avoid division by zero during scoring.")
                  MIN_SIGMA
                }
              valueCol -> finalSigma
            })
        }).toMap
      } finally {
        transformed.unpersist()
      }

      val meta = tmpMeta.copy(trainSigmas = trainSigmas)
      logger.info(s"Generated training metadata}: minTs=${meta.minTs}, rangeTs=${meta.rangeTs}, trainSigmas=${meta.trainSigmas}, cpPositions=${meta.cpPositions.mkString("[", ", ", "]")}")
      logger.info("Model created successfully, ciao!")
      HSModel(pipelineModel, meta)
    }
    finally {
      featured.unpersist()
    }
  }

  /**
   * Convenience method to train the model and store it in one step.
   *
   * @param historicDf DataFrame with historical data for training. Must have columns:
   *                   - timestamp (TimestampType, DateType, StringType with format yyyy-MM-dd HH:mm:ss,
   *                     or LongType/IntegerType interpreted as Unix epoch seconds)
   *                   - one or more numeric metric columns whose names are listed in `valueCols`
   * @param modelPath  path where to save the model and metadata (e.g. "s3://my-bucket/spark-prophet")
   * @param valueCols  set of column names in `historicDf` that contain the numeric metric values to model
   * @param config     HSConfig with Fourier/changepoint hyper-parameters
   * @param spark      implicit SparkSession
   * @return HSModel containing the trained PipelineModel and HSMeta with training parameters and sigmas for scoring
   */
  def fitAndStore(historicDf: DataFrame, modelPath: String, valueCols: Set[String], config: HSConfig = HSConfig())(implicit spark: SparkSession): HSModel = {
    val model = fit(historicDf, valueCols, config)
    store(model, modelPath)
    model
  }

  // ─── PHASE 2: Scoring

  /**
   * Scores a DataFrame with one or more metrics using the trained model.
   * Always uses minTs/rangeTs/cpPositions from training → features identical to the fit.
   * Never accesses historical data → no data leakage.
   *
   * @param model     HSModel containing the trained PipelineModel and HSMeta
   * @param df        DataFrame with new data to score. Must have columns:
   *                       - timestamp (TimestampType, DateType, StringType with format yyyy-MM-dd HH:mm:ss,
   *                         or LongType/IntegerType interpreted as Unix epoch seconds)
   *                       - one or more numeric metric columns whose names are listed in `valueCols`
   * @param valueCols set of metric column names to score (must be a subset of the columns the model was trained on)
   * @param threshold z-score threshold to classify anomalies (e.g. 2.5 means 2.5 standard deviations from the norm; must be > 0)
   * @return DataFrame with all original input columns preserved, plus per-metric scoring columns
   *         prefixed with `{metricName}_`:
   *         - {metric}_forecast: predicted value from the model (rounded to 1 decimal)
   *         - {metric}_residual: value − forecast (rounded to 1 decimal)
   *         - {metric}_z_score: residual / trainSigma (rounded to 2 decimals)
   *         - {metric}_pct_deviation: (value − forecast) / forecast * 100 (rounded to 1 decimal; null when forecast ≈ 0)
   *         - {metric}_is_anomaly: true if abs(z_score) > sigmaThreshold
   *         - {metric}_severity: "HIGH" if abs(z_score) > 2 * threshold, "MEDIUM" if > 1.5 * threshold,
   *           "LOW" if > threshold, "OK" otherwise
   *         - {metric}_minForecast: forecast − sigmaThreshold * trainSigma (lower bound of "normal" range)
   *         - {metric}_maxForecast: forecast + sigmaThreshold * trainSigma (upper bound of "normal" range)
   */
  def predict(model: HSModel, df: DataFrame, valueCols: Set[String], threshold: Double = DEFAULT_Z_THRESHOLD): DataFrame = {

    logger.info("Starting scoring with HelterSkelter model")
    require(threshold > 0, s"sigmaThreshold must be > 0, got $threshold")
    require(valueCols != null && valueCols.nonEmpty, "valueCols must not be empty")
    validateInput(df, Set("timestamp") ++ valueCols, "score")

    val meta = model.meta
    val trainSigmas = meta.trainSigmas

    // Validate that all requested valueCols were present during training
    val unknownCols = valueCols -- trainSigmas.keySet
    require(unknownCols.isEmpty,
      s"[score] valueCols ${unknownCols.mkString(", ")} were not present during training. " +
        s"Trained metrics: ${trainSigmas.keySet.mkString(", ")}")

    val featured = addFeatures(df, meta)
    logger.info("Scoring and classifying anomalies with threshold " + threshold)

    // All computations use the full-precision raw columns (_residual_raw, _z_score_raw);
    // rounded display columns are created at the end, and raw intermediates are dropped.
    val transformed = model.model.transform(featured)
    val scored = valueCols.foldLeft(transformed)((acc, valueCol) =>
      acc.withColumn(valueCol + "_residual_raw", col(valueCol) - col(valueCol + "_prediction"))
        .withColumn(valueCol + "_z_score_raw", col(valueCol + "_residual_raw") / lit(trainSigmas(valueCol)))
        .withColumn(valueCol + "_pct_deviation",
          when(abs(col(valueCol + "_prediction")) > MIN_SIGMA,
            round((col(valueCol) - col(valueCol + "_prediction")) / col(valueCol + "_prediction") * 100, 1))
            .otherwise(lit(null).cast(DoubleType)))
        .withColumn(valueCol + "_is_anomaly", abs(col(valueCol + "_z_score_raw")) > threshold)
        .withColumn(valueCol + "_severity",
          when(abs(col(valueCol + "_z_score_raw")) > threshold * 2.0, "HIGH")
            .when(abs(col(valueCol + "_z_score_raw")) > threshold * 1.5, "MEDIUM")
            .when(abs(col(valueCol + "_z_score_raw")) > threshold, "LOW")
            .otherwise("OK"))
        .withColumn(valueCol + "_minForecast", round(col(valueCol + "_prediction") - lit(threshold * trainSigmas(valueCol)), 1))
        .withColumn(valueCol + "_maxForecast", round(col(valueCol + "_prediction") + lit(threshold * trainSigmas(valueCol)), 1))
        .withColumn(valueCol + "_forecast", round(col(valueCol + "_prediction"), 1))
        .withColumn(valueCol + "_residual", round(col(valueCol + "_residual_raw"), 1))
        .withColumn(valueCol + "_z_score", round(col(valueCol + "_z_score_raw"), 2))
    )

    // Drop intermediate columns added by addFeatures, VectorAssembler, prediction, and raw scoring intermediates
    val colsToDrop = featureColNames(meta) ++ Seq("features") ++
      valueCols.flatMap(vc => Seq(vc + "_prediction", vc + "_residual_raw", vc + "_z_score_raw"))
    val result = scored.drop(colsToDrop: _*)

    logger.info("Scoring completed, ciao!")
    result
  }

  /**
   * Convenience method to load the model and score in one step.
   *
   * @param modelPath path where the model and metadata were saved during training (e.g. "s3://my-bucket/spark-prophet")
   * @param df        DataFrame with new data to score. Must have columns:
   *                       - timestamp (TimestampType, DateType, StringType with format yyyy-MM-dd HH:mm:ss,
   *                         or LongType/IntegerType interpreted as Unix epoch seconds)
   *                       - one or more numeric metric columns whose names are listed in `valueCols`
   * @param valueCols set of metric column names to score (must be a subset of the columns the model was trained on)
   * @param threshold z-score threshold to classify anomalies (e.g. 2.5 means 2.5 standard deviations from the norm; must be > 0)
   * @param spark     implicit SparkSession
   * @return DataFrame with all original input columns preserved, plus the per-metric scoring columns described in scoreWithModel().
   * @see scoreWithModel()
   * @see loadModel()
   * @note This method is less efficient if you need to score multiple DataFrames in a row, since it loads the model from storage every time.
   *       In that case, it's better to call loadModel() once and reuse the HSModel instance for multiple scoreWithModel() calls.
   */
  def loadAndPredict(modelPath: String, df: DataFrame, valueCols: Set[String], threshold: Double = DEFAULT_Z_THRESHOLD)(implicit spark: SparkSession): DataFrame =
    predict(load(modelPath), df, valueCols, threshold)

  // ─── Feature engineering ──────────────────────────────────────────────────────

  /**
   * Adds features to the DataFrame — identical for training and scoring.
   *
   * Internally, an intermediate column `ts_hours` (timestamp in hours from Unix epoch)
   * is computed and used as the base unit for all feature calculations. It is dropped
   * before the method returns and is never present in the output DataFrame.
   *
   * Columns produced (present in the returned DataFrame):
   *
   * trend    : (ts_hours − minTs) / rangeTs → normalized in [0,1] on the training set.
   * For future data it will be > 1 (trend extrapolation).
   * The regression learns the weight w_trend that
   * represents the base slope: each unit of trend corresponds
   * to w_trend bookings more (or fewer if negative).
   *
   * cp_1..N  : changepoint hinge functions (N = nChangepoints from config).
   * cp_j = max(0, ts_hours − s_j) / rangeTs
   * Equals 0 before the changepoint, increases after.
   * The weight δ_j (positive or negative) modifies the trend slope.
   *
   * hourly_sin_k, hourly_cos_k  : hourly Fourier terms (period HOURS_PER_DAY)
   * weekly_sin_k, weekly_cos_k  : weekly Fourier terms (period HOURS_PER_WEEK)
   * monthly_sin_k, monthly_cos_k : monthly Fourier terms (period HOURS_PER_MONTH)
   * yearly_sin_k, yearly_cos_k  : yearly Fourier terms (period HOURS_PER_YEAR)
   *
   * angle = ts_hours / periodHours * 2πk  (pure number, no units)
   * Now everything is consistent: ts_hours / periodHours = completed cycles
   *
   * @param df   input DataFrame; must contain a "timestamp" column of a supported type
   *             (TimestampType, DateType, StringType with format yyyy-MM-dd HH:mm:ss,
   *             or LongType/IntegerType interpreted as Unix epoch seconds).
   *             Any other metric columns are passed through unchanged.
   * @param meta Fourier/changepoint configuration (either HSConfig or HSMeta)
   * @return DataFrame with the original columns plus the added feature columns (trend, cp_*, Fourier terms)
   */
  private def addFeatures(
                           df: DataFrame,
                           meta: HSMeta
                         ): DataFrame = {

    logger.info("Adding features")

    // Compute ts_hours once as a column, then reference it by name in subsequent expressions.
    // toTsHours handles all supported timestamp column types (Timestamp, Date, String, Long, Int).
    val withTsHours = df.withColumn("ts_hours", toTsHours(df))

    // Build all feature columns referencing the materialized ts_hours column.
    // ts_hours is dropped at the end of this method.
    val tNorm = (col("ts_hours") - lit(meta.minTs)) / lit(meta.rangeTs)

    val baseCols: Seq[Column] = Seq(tNorm.as("trend"))

    // Changepoint hinge functions
    val cpCols: Seq[Column] = meta.cpPositions.zipWithIndex.map { case (cpVal, i) =>
      (greatest(col("ts_hours") - lit(cpVal), lit(0.0)) / lit(meta.rangeTs)).as(s"cp_${i + 1}")
    }

    // Fourier terms: ts_hours / periodHours = number of completed cycles
    // Multiplied by 2πk → angle in radians → sin/cos
    def fourierCols(periodHours: Double, order: Int, name: String): Seq[Column] =
      (1 to order).flatMap { k =>
        val angle = col("ts_hours") / lit(periodHours) * lit(2 * Math.PI * k)
        Seq(sin(angle).as(s"${name}_sin_$k"), cos(angle).as(s"${name}_cos_$k"))
      }

    val allNewCols = baseCols ++ cpCols ++
      fourierCols(HOURS_PER_DAY, meta.fourierOrderHourly, "hourly") ++
      fourierCols(HOURS_PER_WEEK, meta.fourierOrderWeekly, "weekly") ++
      fourierCols(HOURS_PER_MONTH, meta.fourierOrderMonthly, "monthly") ++
      fourierCols(HOURS_PER_YEAR, meta.fourierOrderYearly, "yearly")

    // ts_hours is only needed to compute the feature columns; drop it to avoid
    // carrying an unnecessary intermediate column through cached DataFrames and pipelines.
    val result = withTsHours.select(col("*") +: allNewCols: _*).drop("ts_hours")
    logger.info("Features added: " + featureColNames(meta).mkString(", "))
    result
  }

  /**
   * Returns the ordered list of feature column names for a given Fourier/changepoint configuration.
   * Accepts either a [[HSConfig]] or a [[HSMeta]].
   *
   * @param meta configuration providing the Fourier orders and number of changepoints
   * @return sequence of feature column names (e.g. Seq("trend", "cp_1", ..., "hourly_sin_1", ...))
   */
  private def featureColNames(meta: HSFeatureSpec): Seq[String] = {
    val trend = Seq("trend") ++ (1 to meta.nChangepoints).map(i => s"cp_$i")

    def fourierNames(name: String, order: Int): Seq[String] =
      (1 to order).flatMap(k => Seq(s"${name}_sin_$k", s"${name}_cos_$k"))

    trend ++
      fourierNames("hourly", meta.fourierOrderHourly) ++
      fourierNames("weekly", meta.fourierOrderWeekly) ++
      fourierNames("monthly", meta.fourierOrderMonthly) ++
      fourierNames("yearly", meta.fourierOrderYearly)
  }

  /**
   * Validates that the input DataFrame contains the required columns.
   *
   * @param df       DataFrame to validate
   * @param required set of required column names
   * @param context  human-readable context for the error message (e.g. "train", "score")
   */
  private def validateInput(df: DataFrame, required: Set[String], context: String): Unit = {
    val dfColumns = df.columns.toSet
    val missing = required -- dfColumns
    require(missing.isEmpty,
      s"[$context] DataFrame is missing required columns: ${missing.mkString(", ")}. " +
        s"Available: ${dfColumns.mkString(", ")}")
  }

  /**
   * Converts the "timestamp" column of a DataFrame to hours from Unix epoch (as a DoubleType Column expression).
   *
   * Supported input types for the "timestamp" column:
   *   - TimestampType / DateType: converted to epoch seconds via `unix_timestamp()`
   *   - StringType: parsed by `unix_timestamp()` with the default format `yyyy-MM-dd HH:mm:ss`
   *   - LongType / IntegerType: used directly (interpreted as Unix epoch seconds)
   *
   * In all cases, the resulting epoch-seconds value is cast to Double and divided by 3600.
   *
   * @param df the input DataFrame — must contain a "timestamp" column of a supported type
   * @return a Column expression producing the timestamp in hours from Unix epoch (DoubleType)
   * @throws IllegalArgumentException if the "timestamp" column type is unsupported
   */
  private def toTsHours(df: DataFrame): Column = {
    val tsCol = df.schema("timestamp").dataType match {
      case LongType | IntegerType => col("timestamp")
      case TimestampType | DateType | StringType => unix_timestamp(col("timestamp"))
      case other =>
        throw new IllegalArgumentException(
          s"Unsupported type for 'timestamp' column: $other. " +
            "Expected one of: LongType, IntegerType (Unix epoch seconds), " +
            "TimestampType, DateType, StringType (yyyy-MM-dd HH:mm:ss).")
    }

    tsCol.cast(DoubleType) / lit(3600.0)
  }

}
