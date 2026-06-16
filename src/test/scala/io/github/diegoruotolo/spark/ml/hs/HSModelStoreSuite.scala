package io.github.diegoruotolo.spark.ml.hs

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import HSModelStore.{META_DIR, MODEL_DIR}
import HSTestData._
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files

class HSModelStoreSuite extends AnyFunSuite with Matchers with DataFrameSuiteBase {

  // Make the inherited `spark` available as an implicit SparkSession for the methods under test
  private implicit def implicitSpark: SparkSession = spark

  import spark.implicits._

  // ─── Helpers ────────────────────────────────────────────────────────────────

  private val TMP_DIR_NAME = "target/hs_model_store_tmp"
  private val tmpDir = new File(TMP_DIR_NAME)

  /** Creates a temp directory under target/, deleting it first if it already exists. */
  private def createTmpDir(): String = {
    deleteTmpDir()
    Files.createDirectories(tmpDir.toPath)
    tmpDir.getPath.replace("\\", "/")
  }

  private def deleteTmpDir(): Unit = {
    if (tmpDir.exists())
      FileUtils.deleteDirectory(tmpDir)
  }

  private val testMeta = HSMeta(
    fourierOrderHourly = 4,
    fourierOrderWeekly = 3,
    fourierOrderMonthly = 5,
    fourierOrderYearly = 8,
    cpPositions = Seq(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    trainSigmas = Map("bookings" -> 1.0, "searches" -> 2.0),
    minTs = 0.0,
    rangeTs = 72.0
  )

  private def createDummyPipelineModel(): PipelineModel = {
    // Tiny deterministic dataset
    val df = Seq(
      (1.0, 2.0, 3.0),
      (2.0, 3.0, 5.0),
      (3.0, 4.0, 7.0)
    ).toDF("f1", "f2", "label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("f1", "f2"))
      .setOutputCol("features")

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(1)
      .setRegParam(0.0)

    new Pipeline()
      .setStages(Array(assembler, lr))
      .fit(df)
  }

  // ─── store() ──────────────────────────────────────────────────────────────────

  test("store - fails when model is null") {
    a[Exception] should be thrownBy {
      HSModelStore.store(null, TMP_DIR_NAME)
    }
  }

  test("store - fails when model pipeline model is null") {
    a[Exception] should be thrownBy {
      HSModelStore.store(HSModel(null, testMeta), TMP_DIR_NAME)
    }
  }

  test("store - fails when metadata are null") {
    a[Exception] should be thrownBy {
      HSModelStore.store(HSModel(createDummyPipelineModel(), null), TMP_DIR_NAME)
    }
  }

  test("store - fails when model path is null") {
    a[Exception] should be thrownBy {
      HSModelStore.store(HSModel(createDummyPipelineModel(), testMeta), null)
    }
  }

  test("store - fails when model path is empty") {
    a[Exception] should be thrownBy {
      HSModelStore.store(HSModel(createDummyPipelineModel(), testMeta), "")
    }
  }

  test("store - successfully saves model + default metadata") {

    createTmpDir()
    HSModelStore.store(HSModel(createDummyPipelineModel(), testMeta), TMP_DIR_NAME)

    // Model and meta directories should have been created
    val modelDir = new java.io.File(TMP_DIR_NAME + MODEL_DIR)
    val metaDir = new java.io.File(TMP_DIR_NAME + META_DIR)
    modelDir.exists() should be(true)
    metaDir.exists() should be(true)

    val resultMeta = spark.read.parquet(TMP_DIR_NAME + META_DIR).as[HSMeta].head()
    resultMeta should equal(testMeta)

    deleteTmpDir()
  }

  // ─── load() ──────────────────────────────────────────────────────────────

  test("load - fails when model path is null") {
    a[Exception] should be thrownBy {
      HSModelStore.load(null)
    }
  }

  test("load - fails when model path does not exist") {
    a[Exception] should be thrownBy {
      HSModelStore.load("/nonexistent/path/that/does/not/exist")
    }
  }

  test("load - successfully loads saved model and metadata") {

    val config = HSConfig()
    val loaded = loadTestModel()

    loaded should not be null
    loaded.model shouldBe a[PipelineModel]
    loaded.meta should not be null
    loaded.meta.fourierOrderHourly should be(config.fourierOrderHourly)
    loaded.meta.fourierOrderWeekly should be(config.fourierOrderWeekly)
    loaded.meta.fourierOrderMonthly should be(config.fourierOrderMonthly)
    loaded.meta.fourierOrderYearly should be(config.fourierOrderYearly)
    loaded.meta.nChangepoints should be(config.nChangepoints)
    loaded.meta.trainSigmas.keys should contain theSameElementsAs testMetrics
    loaded.meta.trainSigmas("bookings") should be > 0.0
    loaded.meta.trainSigmas("searches") should be > 0.0
    loaded.meta.rangeTs should be > 0.0
  }

}
