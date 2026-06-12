package io.rolling.spark.ml.hs

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

class HelterSkelterSuite extends AnyFunSuite with Matchers with DataFrameSuiteBase {

  // Make the inherited `spark` available as an implicit SparkSession for the methods under test
  private implicit def implicitSpark: SparkSession = spark
  import spark.implicits._

  // ─── Helpers ────────────────────────────────────────────────────────────────

  private val TMP_DIR_NAME = "sparkly_prophet_test"
  private val START_TIME = LocalDateTime.of(2025, 1, 1, 0, 0, 0)
  private val dtFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")

  /** Creates a temp directory under target/, deleting it first if it already exists. */
  private def createTmpDir(): String = {
    val dir = new File("target", TMP_DIR_NAME)
    if (dir.exists())
      FileUtils.deleteDirectory(dir)
    Files.createDirectories(dir.toPath)
    dir.getPath.replace("\\", "/")
  }

  /** Builds a small synthetic training DataFrame: 72 hours (3 days) of hourly data.
   * bookings ≈ 100 + hour + noise, searches ≈ 500 + 3 * hour + noise.
   * A deterministic pseudo-random noise is added so that the training sigma is
   * realistic (not near-zero) while keeping the test fully reproducible. */
  private def buildTrainingDf(): DataFrame = {
    val rng = new scala.util.Random(42) // fixed seed for reproducibility
    val rows = (0 until 72).map { hour =>
      val ts = START_TIME.plusHours(hour).format(dtFormatter)
      val bookings = 100 + hour + rng.nextInt(21) - 10   // ±10 noise
      val searches = 500 + 3 * hour + rng.nextInt(61) - 30 // ±30 noise
      (ts, bookings, searches)
    }
    rows.toDF("timestamp", "bookings", "searches")
  }

  /** Builds a small scoring DataFrame (next 6 hours after training).
   * Values follow the same base trend as training but without noise. */
  private def buildScoringDf(): DataFrame = {
    val rows = (72 until 78).map { hour =>
      val ts = START_TIME.plusHours(hour).format(dtFormatter)
      val bookings = 100 + hour
      val searches = 500 + 3 * hour
      (ts, bookings, searches)
    }
    rows.toDF("timestamp", "bookings", "searches")
  }

  // ─── trainAndStore() ──────────────────────────────────────────────────────────────────

  test("train - successfully trains and saves model + metadata") {
    val modelPath = createTmpDir()
    val df = buildTrainingDf()

    HelterSkelter.trainAndStore(df, modelPath, Set("bookings", "searches"))

    // Model and meta directories should have been created
    val modelDir = new java.io.File(modelPath + HelterSkelter.MODEL_DIR)
    val metaDir = new java.io.File(modelPath + HelterSkelter.META_DIR)
    modelDir.exists() should be(true)
    metaDir.exists() should be(true)
  }

  test("train - saves correct metadata") {
    val modelPath = createTmpDir()
    val df = buildTrainingDf()
    val config = HSConfig(
      fourierOrderHourly = 2,
      fourierOrderWeekly = 1,
      fourierOrderMonthly = 0,
      fourierOrderYearly = 0,
      nChangepoints = 3
    )

    HelterSkelter.trainAndStore(df, modelPath, Set("bookings"), config)
    val meta = spark.read.parquet(modelPath + HelterSkelter.META_DIR)
      .as[HSMeta].head()

    meta.fourierOrderHourly should be(2)
    meta.fourierOrderWeekly should be(1)
    meta.fourierOrderMonthly should be(0)
    meta.fourierOrderYearly should be(0)
    meta.nChangepoints should be(3)
    meta.trainSigmas should contain key "bookings"
    meta.trainSigmas("bookings") should be > 0.0
    meta.rangeTs should be > 0.0
  }

  test("train - fails with empty valueCols") {
    val df = buildTrainingDf()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set.empty[String])
    }
  }

  test("train - fails with null valueCols") {
    val df = buildTrainingDf()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, null)
    }
  }

  test("train - fails when valueCols column is missing from DataFrame") {
    val df = buildTrainingDf()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set("nonexistent_column"))
    }
  }

  test("train - fails when timestamp column is missing from DataFrame") {
    val df = Seq((1.0, 2.0)).toDF("bookings", "searches")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set("bookings"))
    }
  }

  test("train - fails on empty DataFrame") {
    val df = Seq.empty[(String, Double)].toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set("bookings"))
    }
  }

  test("train - fails when all timestamps are identical (zero range)") {
    val df = (1 to 5).map(_ => ("2025-01-01 00:00:00", 100.0)).toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set("bookings"))
    }
  }

  test("train - supports LongType timestamp column") {
    val baseEpoch = 1735689600L // 2025-01-01 00:00:00 UTC
    val rows = (0 until 72).map { h =>
      val ts = baseEpoch + h * 3600L
      val bookings = 100.0 + 2.0 * h
      (ts, bookings)
    }
    val df = rows.toDF("timestamp", "bookings")

    // Should not throw
    HelterSkelter.train(df, Set("bookings")) should not be null
  }

  test("train - supports IntegerType timestamp column") {
    val baseEpoch = 1735689600 // fits in Int for 2025
    val rows = (0 until 72).map { h =>
      val ts = baseEpoch + h * 3600
      val bookings = 100.0 + 2.0 * h
      (ts, bookings)
    }
    val df = rows.toDF("timestamp", "bookings")

    // Should not throw
    HelterSkelter.train(df, Set("bookings")) should not be null
  }

  test("train - fails on unsupported timestamp column type (BooleanType)") {
    val df = Seq((true, 100.0), (false, 200.0)).toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.train(df, Set("bookings"))
    }
  }

  test("train - with zero changepoints") {
    val df = buildTrainingDf()
    val config = HSConfig(nChangepoints = 0)

    // Should not throw
    val model = HelterSkelter.train(df, Set("bookings"), config)

    model should not be null
    model.meta should not be null
    model.meta.nChangepoints should be(0)
    model.meta.cpPositions should be(empty)
  }

  // ─── loadModel() ──────────────────────────────────────────────────────────────

  test("loadModel - successfully loads saved model and metadata") {
    val modelPath = createTmpDir()
    val df = buildTrainingDf()
    val config = HSConfig(fourierOrderHourly = 3, fourierOrderWeekly = 2, nChangepoints = 5)

    HelterSkelter.trainAndStore(df, modelPath, Set("bookings", "searches"), config)

    val loaded = HelterSkelter.loadModel(modelPath)

    loaded should not be null
    loaded.model shouldBe a[PipelineModel]
    loaded.meta.fourierOrderHourly should be(3)
    loaded.meta.fourierOrderWeekly should be(2)
    loaded.meta.nChangepoints should be(5)
    loaded.meta.trainSigmas should contain key "bookings"
    loaded.meta.trainSigmas should contain key "searches"
  }

  test("loadModel - fails when model path does not exist") {
    a[Exception] should be thrownBy {
      HelterSkelter.loadModel("/nonexistent/path/that/does/not/exist")
    }
  }

  test("loadModel - preserves minTs and rangeTs from training") {
    val modelPath = createTmpDir()
    val df = buildTrainingDf()

    HelterSkelter.trainAndStore(df, modelPath, Set("bookings"))

    val loaded = HelterSkelter.loadModel(modelPath)
    loaded.meta.minTs should be > 0.0
    loaded.meta.rangeTs should be > 0.0
  }

  // ─── scoreWithModel() ─────────────────────────────────────────────────────────

  test("scoreWithModel - produces all expected output columns") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val loaded = HelterSkelter.loadModel(modelPath)
    val scoreDf = buildScoringDf()

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"), 2.5)

    val resultCols = result.columns.toSet
    // Original columns preserved
    resultCols should contain("timestamp")
    resultCols should contain("bookings")
    resultCols should contain("searches") // passed through unchanged

    // Scoring columns for "bookings"
    resultCols should contain("bookings_forecast")
    resultCols should contain("bookings_residual")
    resultCols should contain("bookings_z_score")
    resultCols should contain("bookings_pct_deviation")
    resultCols should contain("bookings_is_anomaly")
    resultCols should contain("bookings_severity")
    resultCols should contain("bookings_minForecast")
    resultCols should contain("bookings_maxForecast")

    // Intermediate columns should NOT be present
    resultCols should not contain "bookings_prediction"
    resultCols should not contain "bookings_residual_raw"
    resultCols should not contain "bookings_z_score_raw"
    resultCols should not contain "features"
    resultCols should not contain "ts_hours"
    resultCols should not contain "trend"
  }

  test("scoreWithModel - returns correct number of rows") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val loaded = HelterSkelter.loadModel(modelPath)
    val scoreDf = buildScoringDf()

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"))
    result.count() should be(scoreDf.count())
  }

  test("scoreWithModel - severity values are within expected set") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val loaded = HelterSkelter.loadModel(modelPath)
    val scoreDf = buildScoringDf()

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"))
    val severities = result.select("bookings_severity").distinct().collect().map(_.getString(0)).toSet
    severities.subsetOf(Set("OK", "LOW", "MEDIUM", "HIGH")) should be(true)
  }

  test("scoreWithModel - scores multiple metrics in one pass") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings", "searches"))

    val loaded = HelterSkelter.loadModel(modelPath)
    val scoreDf = buildScoringDf()

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings", "searches"))

    val resultCols = result.columns.toSet
    resultCols should contain("bookings_forecast")
    resultCols should contain("searches_forecast")
    resultCols should contain("bookings_z_score")
    resultCols should contain("searches_z_score")
  }

  test("scoreWithModel - can score a subset of trained metrics") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings", "searches"))

    val loaded = HelterSkelter.loadModel(modelPath)
    val scoreDf = buildScoringDf()

    // Score only "bookings" even though model was trained on both
    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"))
    result.columns.toSet should contain("bookings_forecast")
    result.columns.toSet should not contain "searches_forecast"
  }

  test("scoreWithModel - fails with empty valueCols") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf(), Set.empty[String])
    }
  }

  test("scoreWithModel - fails with null valueCols") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf(), null)
    }
  }

  test("scoreWithModel - fails when valueCols are not present in scoring DataFrame") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    val ex = the[IllegalArgumentException] thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf(), Set("nonexistent_metric"))
    }
    ex.getMessage should include("DataFrame is missing required columns: nonexistent_metric")
  }

  test("scoreWithModel - fails when valueCols were not present during training") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    val ex = the[IllegalArgumentException] thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf().withColumn("nonexistent_metric", lit(null)), Set("nonexistent_metric"))
    }
    ex.getMessage should include("were not present during training")
  }

  test("scoreWithModel - fails when sigmaThreshold is zero") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf(), Set("bookings"), sigmaThreshold = 0.0)
    }
  }

  test("scoreWithModel - fails when sigmaThreshold is negative") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.scoreWithModel(loaded)(buildScoringDf(), Set("bookings"), sigmaThreshold = -1.0)
    }
  }

  test("scoreWithModel - fails when timestamp column is missing from scoring DataFrame") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)
    val badDf = Seq(100.0).toDF("bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.scoreWithModel(loaded)(badDf, Set("bookings"))
    }
  }

  test("scoreWithModel - detects injected anomaly") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    // Build scoring data with one extreme outlier
    val normalRows = (72 until 77).map { hour =>
      (START_TIME.plusHours(hour).format(dtFormatter), 100 + hour)
    }
    val allRows = normalRows :+ (START_TIME.plusHours(77).format(dtFormatter), 99999) // extreme anomaly
    val scoreDf = allRows.toDF("timestamp", "bookings")

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"))
    val anomalies = result.filter("bookings_is_anomaly = true")
    anomalies.count() should be (1L)
  }

  test("scoreWithModel - supports LongType timestamp in scoring DataFrame") {
    val baseEpoch = 1735689600L // 2025-01-01 00:00:00 UTC
    val trainRows = (0 until 72).map { h =>
      (baseEpoch + h * 3600L, 100.0 + 2.0 * h)
    }
    val trainDf = trainRows.toDF("timestamp", "bookings")
    val modelPath = createTmpDir()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))
    val loaded = HelterSkelter.loadModel(modelPath)

    val scoreRows = (72 until 78).map { h =>
      (baseEpoch + h * 3600L, 100.0 + 2.0 * h)
    }
    val scoreDf = scoreRows.toDF("timestamp", "bookings")

    val result = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"))
    result.count() should be(6)
    result.columns.toSet should contain("bookings_forecast")
  }

  // ─── score() (convenience method) ─────────────────────────────────────────────

  test("score - loads model and scores in one step") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val scoreDf = buildScoringDf()
    val result = HelterSkelter.score(modelPath, scoreDf, Set("bookings"))

    result.count() should be(scoreDf.count())
    result.columns.toSet should contain("bookings_forecast")
    result.columns.toSet should contain("bookings_is_anomaly")
  }

  test("score - fails with empty valueCols") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.score(modelPath, buildScoringDf(), Set.empty[String])
    }
  }

  test("score - fails with null valueCols") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.score(modelPath, buildScoringDf(), null)
    }
  }

  test("score - fails when model path does not exist") {
    a[Exception] should be thrownBy {
      HelterSkelter.score("/nonexistent/model/path", buildScoringDf(), Set("bookings"))
    }
  }

  test("score - fails when valueCols not in DataFrame") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val ex = the[IllegalArgumentException] thrownBy {
      HelterSkelter.score(modelPath, buildScoringDf(), Set("unknown_metric"))
    }
    ex.getMessage should include("DataFrame is missing required columns: unknown_metric")
  }

  test("score - fails when valueCols not in training") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val ex = the[IllegalArgumentException] thrownBy {
      HelterSkelter.score(modelPath, buildScoringDf().withColumn("unknown_metric", lit(null)), Set("unknown_metric"))
    }
    ex.getMessage should include("valueCols unknown_metric were not present during training")
  }

  test("score - produces same results as loadModel + scoreWithModel") {
    val modelPath = createTmpDir()
    val trainDf = buildTrainingDf()
    HelterSkelter.trainAndStore(trainDf, modelPath, Set("bookings"))

    val scoreDf = buildScoringDf()
    val resultViaScore = HelterSkelter.score(modelPath, scoreDf, Set("bookings"), sigmaThreshold = 3.0)
    val loaded = HelterSkelter.loadModel(modelPath)
    val resultViaScoreWithModel = HelterSkelter.scoreWithModel(loaded)(scoreDf, Set("bookings"), sigmaThreshold = 3.0)

    // Same columns
    resultViaScore.columns.sorted should be(resultViaScoreWithModel.columns.sorted)

    // Same row count
    resultViaScore.count() should be(resultViaScoreWithModel.count())

    // Same forecast values (collect and compare)
    val forecastsA = resultViaScore.select("bookings_forecast").collect().map(_.getDouble(0))
    val forecastsB = resultViaScoreWithModel.select("bookings_forecast").collect().map(_.getDouble(0))
    forecastsA should be(forecastsB)
  }

}

