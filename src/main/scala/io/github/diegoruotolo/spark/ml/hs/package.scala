package io.github.diegoruotolo.spark.ml.hs

import org.apache.spark.ml.PipelineModel


/**
 * Common trait for types that carry the Fourier/changepoint configuration
 * needed to compute feature column names.
 *
 * Implemented by both [[HSConfig]] (used during training) and
 * [[HSMeta]] (loaded from storage during scoring), so that
 * [[HelterSkelter.featureColNames]] can accept either one.
 */
trait HSFeatureSpec {
  def fourierOrderHourly: Int

  def fourierOrderWeekly: Int

  def fourierOrderMonthly: Int

  def fourierOrderYearly: Int

  def nChangepoints: Int
}

/**
 * Configuration parameters for the HelterSkelter anomaly detection model.
 *
 * @param fourierOrderHourly  number of Fourier terms for hourly seasonality (period 24h)
 * @param fourierOrderWeekly  number of Fourier terms for weekly seasonality (period 168h)
 * @param fourierOrderMonthly number of Fourier terms for monthly seasonality (period 720.5h ≈ 30.02 days)
 * @param fourierOrderYearly  number of Fourier terms for yearly seasonality (period 8766h)
 * @param nChangepoints       number of changepoints to use in the piecewise linear trend (equidistant in the training period)
 */
case class HSConfig(
                     fourierOrderHourly: Int = 4, // 24h cycle:    captures morning/evening peaks
                     fourierOrderWeekly: Int = 3, // 168h cycle:   weekday vs weekend
                     fourierOrderMonthly: Int = 5, // 720.5h cycle: monthly pattern (≈30.02 days)
                     fourierOrderYearly: Int = 8, // 8766h cycle:  summer vs winter
                     nChangepoints: Int = 9 // equidistant changepoints in training
                   ) extends HSFeatureSpec {
  require(fourierOrderHourly >= 0, s"fourierOrderHourly must be >= 0, got $fourierOrderHourly")
  require(fourierOrderWeekly >= 0, s"fourierOrderWeekly must be >= 0, got $fourierOrderWeekly")
  require(fourierOrderMonthly >= 0, s"fourierOrderMonthly must be >= 0, got $fourierOrderMonthly")
  require(fourierOrderYearly >= 0, s"fourierOrderYearly must be >= 0, got $fourierOrderYearly")
  require(nChangepoints >= 0, s"nChangepoints must be >= 0, got $nChangepoints")
}

/**
 * Metadata computed during training and reused unchanged during scoring.
 *
 * Defined at package level (not as an inner class) so that Spark's implicit
 * Encoder derivation works reliably across all Spark/Scala versions.
 *
 * @param minTs               timestamp of the first training record, in HOURS from Unix epoch.
 *                            Serves as "zero" to normalize trend.
 * @param rangeTs             total duration of the training set in HOURS.
 *                            Used to normalize trend in [0,1] and to scale changepoint hinges.
 * @param trainSigmas         per-metric standard deviation of residuals (observed − predicted) on the training set.
 *                            Keys are the value column names used during training; values are the corresponding sigmas.
 *                            Each sigma is the unit of measure to decide if a residual for that metric is "normal" or anomalous.
 * @param cpPositions         changepoint positions in HOURS from Unix epoch.
 *                            Must be the same in training and scoring, otherwise the hinges
 *                            would have different values and the learned weights would not match.
 *                            Stored for config-compatibility validation at scoring time.
 * @param fourierOrderHourly  Fourier order for hourly seasonality used during training.
 * @param fourierOrderWeekly  Fourier order for weekly seasonality used during training.
 * @param fourierOrderMonthly Fourier order for monthly seasonality used during training.
 * @param fourierOrderYearly  Fourier order for yearly seasonality used during training.
 */
case class HSMeta(
                   minTs: Double,
                   rangeTs: Double,
                   trainSigmas: Map[String, Double],
                   cpPositions: Seq[Double],
                   fourierOrderHourly: Int,
                   fourierOrderWeekly: Int,
                   fourierOrderMonthly: Int,
                   fourierOrderYearly: Int
                 ) extends HSFeatureSpec {
  override def nChangepoints: Int = cpPositions.size
}

/**
 * Container for the trained model and its associated metadata.
 *
 * @param model trained PipelineModel containing a shared VectorAssembler followed by
 *              one LinearRegressionModel per metric (value column)
 * @param meta  HSMeta with the training configuration and parameters needed for scoring
 */
case class HSModel(model: PipelineModel, meta: HSMeta)

/**
 * Custom exception for errors related to the HelterSkelter model.
 *
 * @param message error message describing the issue
 * @param cause   optional underlying cause of the exception (e.g. from a failed model load)
 */
class ModelException(message: String, cause: Throwable = null) extends Exception(message, cause)

