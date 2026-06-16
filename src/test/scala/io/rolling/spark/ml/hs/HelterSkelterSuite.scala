package io.rolling.spark.ml.hs

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import io.rolling.spark.ml.hs.HSTestData._
import io.rolling.spark.ml.hs.HelterSkelter.DEFAULT_Z_THRESHOLD
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class HelterSkelterSuite extends AnyFunSuite with Matchers with DataFrameSuiteBase {

  import spark.implicits._

  // Make the inherited `spark` available as an implicit SparkSession for the methods under test
  private implicit def implicitSpark: SparkSession = spark

  // ─── Helpers ────────────────────────────────────────────────────────────────
  private def doPredict(df: DataFrame = buildDfToPredict(), valueCols: Set[String] = testMetrics, sigmaThreshold: Double = DEFAULT_Z_THRESHOLD): DataFrame =
    HelterSkelter.loadAndPredict(testModelPath, df, valueCols, sigmaThreshold)

  // ─── fit() ──────────────────────────────────────────────────────────────────

  test("fit - fails with empty valueCols") {
    val df = buildDfToTrain()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set.empty[String])
    }
  }

  test("fit - fails with null valueCols") {
    val df = buildDfToTrain()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, null)
    }
  }

  test("fit - fails when valueCols column is missing from DataFrame") {
    val df = buildDfToTrain()
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set("nonexistent_column"))
    }
  }

  test("fit - fails when timestamp column is missing from DataFrame") {
    val df = Seq((1.0, 2.0)).toDF("bookings", "searches")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set("bookings"))
    }
  }

  test("fit - fails on empty DataFrame") {
    val df = Seq.empty[(String, Double)].toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set("bookings"))
    }
  }

  test("fit - fails when all timestamps are identical (zero range)") {
    val df = (1 to 5).map(_ => ("2025-01-01 00:00:00", 100.0)).toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set("bookings"))
    }
  }

  test("fit - supports LongType timestamp column") {
    val baseEpoch = 1735689600L // 2025-01-01 00:00:00 UTC
    val rows = (0 until 72).map { h =>
      val ts = baseEpoch + h * 3600L
      val bookings = 100.0 + 2.0 * h
      (ts, bookings)
    }
    val df = rows.toDF("timestamp", "bookings")

    // Should not throw
    HelterSkelter.fit(df, Set("bookings")) should not be null
  }

  test("fit - supports IntegerType timestamp column") {
    val baseEpoch = 1735689600 // fits in Int for 2025
    val rows = (0 until 72).map { h =>
      val ts = baseEpoch + h * 3600
      val bookings = 100.0 + 2.0 * h
      (ts, bookings)
    }
    val df = rows.toDF("timestamp", "bookings")

    // Should not throw
    HelterSkelter.fit(df, Set("bookings")) should not be null
  }

  test("fit - fails on unsupported timestamp column type (BooleanType)") {
    val df = Seq((true, 100.0), (false, 200.0)).toDF("timestamp", "bookings")
    an[IllegalArgumentException] should be thrownBy {
      HelterSkelter.fit(df, Set("bookings"))
    }
  }

  test("fit - with zero changepoints") {
    val df = buildDfToTrain()
    val config = HSConfig(nChangepoints = 0)

    // Should not throw
    val model = HelterSkelter.fit(df, Set("bookings"), config)

    model should not be null
    model.meta should not be null
    model.meta.nChangepoints should be(0)
    model.meta.cpPositions should be(empty)
  }

  test("fit - correct model") {

    val expected = loadTestModel()
    val result = HelterSkelter.fit(buildDfToTrain(), testMetrics)

    result should not be null

    result.model should not be null
    result.model shouldBe a[PipelineModel]

    result.meta.fourierOrderHourly should be(expected.meta.fourierOrderHourly)
    result.meta.fourierOrderWeekly should be(expected.meta.fourierOrderWeekly)
    result.meta.fourierOrderMonthly should be(expected.meta.fourierOrderMonthly)
    result.meta.fourierOrderYearly should be(expected.meta.fourierOrderYearly)
    result.meta.cpPositions should contain theSameElementsAs expected.meta.cpPositions
    result.meta.rangeTs should be (expected.meta.rangeTs)
    result.meta.trainSigmas.keys should contain theSameElementsAs expected.meta.trainSigmas.keys
    result.meta.trainSigmas("bookings") should be (expected.meta.trainSigmas("bookings") +- 0.01)
    result.meta.trainSigmas("searches") should be (expected.meta.trainSigmas("searches") +- 0.01)
  }

  test("fit - saves correct metadata") {
    val config = HSConfig(
      fourierOrderHourly = 2,
      fourierOrderWeekly = 1,
      fourierOrderMonthly = 0,
      fourierOrderYearly = 0,
      nChangepoints = 3
    )

    val model = HelterSkelter.fit(buildDfToTrain(), testMetrics, config)
    val meta = model.meta

    meta.fourierOrderHourly should be(config.fourierOrderHourly)
    meta.fourierOrderWeekly should be(config.fourierOrderWeekly)
    meta.fourierOrderMonthly should be(config.fourierOrderMonthly)
    meta.fourierOrderYearly should be(config.fourierOrderYearly)
    meta.nChangepoints should be(config.nChangepoints)
    meta.trainSigmas.keys should contain theSameElementsAs testMetrics
    meta.trainSigmas("bookings") should be > 0.0
    meta.trainSigmas("searches") should be > 0.0
    meta.rangeTs should be > 0.0
  }

  // ─── predict() ─────────────────────────────────────────────────────────

  test("predict - fails with empty valueCols") {
    an[IllegalArgumentException] should be thrownBy {
      doPredict(valueCols = Set.empty[String])
    }
  }

  test("predict - fails with null valueCols") {
    an[IllegalArgumentException] should be thrownBy {
      doPredict(valueCols = null)
    }
  }

  test("predict - fails when valueCols are not present in scoring DataFrame") {
    val ex = the[IllegalArgumentException] thrownBy {
      doPredict(valueCols = Set("nonexistent_metric"))
    }
    ex.getMessage should include("DataFrame is missing required columns: nonexistent_metric")
  }

  test("predict - fails when valueCols were not present during training") {
    val ex = the[IllegalArgumentException] thrownBy {
      doPredict(df = buildDfToPredict().withColumn("nonexistent_metric", lit(null)), valueCols = Set("nonexistent_metric"))
    }
    ex.getMessage should include("were not present during training")
  }

  test("predict - fails when sigmaThreshold is zero") {
    an[IllegalArgumentException] should be thrownBy {
      doPredict(sigmaThreshold = 0.0)
    }
  }

  test("predict - fails when sigmaThreshold is negative") {
    an[IllegalArgumentException] should be thrownBy {
      doPredict(sigmaThreshold = -1.0)
    }
  }

  test("predict - fails when timestamp column is missing from scoring DataFrame") {
    val badDf = Seq(100.0).toDF("bookings")
    an[IllegalArgumentException] should be thrownBy {
      doPredict(df = badDf)
    }
  }

  test("predict - checks output columns and row count") {

    val dfToPredict = buildDfToPredict()
    val result = doPredict(df = dfToPredict)

    // Columns check
    val expectedCols = Set(
      "timestamp",
      "bookings", "searches",
      "bookings_forecast", "bookings_residual", "bookings_z_score", "bookings_pct_deviation", "bookings_is_anomaly", "bookings_severity", "bookings_minForecast", "bookings_maxForecast",
      "searches_forecast", "searches_residual", "searches_z_score", "searches_pct_deviation", "searches_is_anomaly", "searches_severity", "searches_minForecast", "searches_maxForecast"
    )
    result.columns.toSet should contain theSameElementsAs expectedCols

    // Number of rows should be correct
    result.count() should be(dfToPredict.count())

    // Severity values are within expected set
    val severities = result.select("bookings_severity").distinct().collect().map(_.getString(0)).toSet
    severities.subsetOf(Set("OK", "LOW", "MEDIUM", "HIGH")) should be(true)
  }


  test("predict - detects injected anomaly") {

    // Build scoring data with one extreme outlier
    val normalRows = (72 until 77).map { hour =>
      (START_TIME.plusHours(hour).format(dtFormatter), 100 + hour)
    }
    val allRows = normalRows :+ (START_TIME.plusHours(77).format(dtFormatter), 99999) // extreme anomaly
    val scoreDf = allRows.toDF("timestamp", "bookings")

    val result = doPredict(df = scoreDf, valueCols = Set("bookings"))
    val anomalies = result.filter("bookings_is_anomaly = true")
    anomalies.count() should be(1L)
  }

  test("predict - predict with model produces same results as predict with model path") {

    val dfToPredict = buildDfToPredict()
    val resultWithModel = HelterSkelter.predict(loadTestModel(), dfToPredict, testMetrics)
    val resultWithPath = HelterSkelter.loadAndPredict(testModelPath, dfToPredict, testMetrics)

    // Same columns
    resultWithModel.columns.sorted should be(resultWithPath.columns.sorted)

    // Same row count
    resultWithModel.count() should be(resultWithPath.count())

    // Same forecast values (collect and compare)
    val forecastsA = resultWithModel.select("bookings_forecast").collect().map(_.getDouble(0))
    val forecastsB = resultWithPath.select("bookings_forecast").collect().map(_.getDouble(0))
    forecastsA should be(forecastsB)
  }

}

