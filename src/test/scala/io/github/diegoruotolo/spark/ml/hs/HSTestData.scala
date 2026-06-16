package io.github.diegoruotolo.spark.ml.hs

import org.apache.spark.sql.{DataFrame, SparkSession}

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Utility object for generating synthetic test data and loading a pre-trained model for unit tests
 */
object HSTestData {

  /**
   * The starting timestamp for the synthetic training data, set to January 1, 2025, at 00:00:00.
   */
  private[hs] val START_TIME = LocalDateTime.of(2025, 1, 1, 0, 0, 0)

  /**
   * DateTimeFormatter to format timestamps in the pattern "yyyy-MM-dd HH:mm:ss".
   */
  private[hs] val dtFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")

  /**
   * Path to the pre-trained model used for testing.
   */
  private[hs] val testModelPath = "src/test/resources/hsmodel"

  /**
   * Set of metrics used in the synthetic test data.
   */
  private[hs] val testMetrics = Set("bookings", "searches")

  /**
   * Builds a small synthetic training DataFrame: 72 hours (3 days) of hourly data.
   * bookings ≈ 100 + hour + noise, searches ≈ 500 + 3 * hour + noise.
   * A deterministic pseudo-random noise is added so that the training sigma is
   * realistic (not near-zero) while keeping the test fully reproducible.
   *
   * @param spark implicit SparkSession
   * @return DataFrame with 72 rows of hourly data for training
   */
  private[hs] def buildDfToTrain()(implicit spark: SparkSession): DataFrame = {
    val rng = new scala.util.Random(42) // fixed seed for reproducibility
    val rows = (0 until 72).map { hour =>
      val ts = START_TIME.plusHours(hour).format(dtFormatter)
      val bookings = 100 + hour + rng.nextInt(21) - 10 // ±10 noise
      val searches = 500 + 3 * hour + rng.nextInt(61) - 30 // ±30 noise
      (ts, bookings, searches)
    }

    import spark.implicits._
    rows.toDF("timestamp", "bookings", "searches")
  }

  /**
   * Builds a small scoring DataFrame (next 6 hours after training).
   * Values follow the same base trend as training but without noise.
   *
   * @param spark implicit SparkSession
   * @return DataFrame with 6 rows of hourly data for scoring
   */
  private[hs] def buildDfToPredict()(implicit spark: SparkSession): DataFrame = {
    val rows = (72 until 78).map { hour =>
      val ts = START_TIME.plusHours(hour).format(dtFormatter)
      val bookings = 100 + hour
      val searches = 500 + 3 * hour
      (ts, bookings, searches)
    }

    import spark.implicits._
    rows.toDF("timestamp", "bookings", "searches")
  }

  /**
   * Loads the pre-trained HSModel from the specified path for use in unit tests.
   *
   * @param spark implicit SparkSession
   * @return HSModel containing the trained PipelineModel and HSMeta
   */
  private[hs] def loadTestModel()(implicit spark: SparkSession): HSModel = {
    HSModelStore.load(testModelPath)
  }

  /**
   * Main method to build the synthetic training DataFrame and fit/store the model for testing.
   * @param args command-line arguments (not used)
   */
  def main(args: Array[String]): Unit = {
    println("Creating Spark session...")
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("HSTestData")
      .master("local[*]")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    println("Creating dataframe to train...")
    val df = buildDfToTrain()
    println("Training and storing dataframe...")
    HelterSkelter.fitAndStore(df, testModelPath, testMetrics)
    println("...done!")
  }

}
