package io.github.diegoruotolo.spark.ml.hs

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

import scala.util.{Failure, Success, Try}

/**
 * Utility object for saving and loading the HSModel (PipelineModel + HSMeta) to/from persistent storage.
 */
object HSModelStore {

  /**
   * Instance of the logger class to correctly log at execution time
   */
  @transient protected lazy val logger: Logger = LogManager.getLogger(getClass)

  val MODEL_DIR = "/model"
  val META_DIR = "/meta"

  /**
   * Saves the given HSModel (PipelineModel + HSMeta) to persistent storage.
   *
   * @param model     the HSModel containing the trained PipelineModel and HSMeta to save
   * @param modelPath path where to save the model and metadata (e.g. "s3://my-bucket/spark-prophet")
   * @param spark     implicit SparkSession
   */
  def store(model: HSModel, modelPath: String)(implicit spark: SparkSession): Unit = {
    require(model != null, "model must not be null")
    require(model.model != null, "trained model must not be null")
    require(model.meta != null, "metadata must not be null")
    require(modelPath != null && modelPath.nonEmpty, "model path must not be null nor empty")

    logger.info(s"Saving model to ${modelPath + MODEL_DIR}")
    model.model.write.overwrite().save(modelPath + MODEL_DIR)
    logger.info(s"Saving training metadata to ${modelPath + META_DIR}")
    spark.createDataFrame(Seq(model.meta)).write.mode("overwrite").parquet(modelPath + META_DIR)
    logger.info("Model and metadata saved successfully, ciao!")
  }

  /**
   * Loads the trained model and metadata from storage.
   *
   * @param modelPath path where the model and metadata were saved during training (e.g. "s3://my-bucket/spark-prophet")
   * @param spark     implicit SparkSession
   * @return HSModel containing the PipelineModel and HSMeta needed for scoring
   */
  def load(modelPath: String)(implicit spark: SparkSession): HSModel = {
    logger.info(s"Loading model from ${modelPath + MODEL_DIR}")
    val model = Try(PipelineModel.load(modelPath + MODEL_DIR)) match {
      case Success(m) => m
      case Failure(e) => throw new ModelException(s"Failed to load model from path ${modelPath + MODEL_DIR}: ${e.getMessage}", e)
    }

    import spark.implicits._
    logger.info(s"Loading metadata from ${modelPath + META_DIR}")
    val meta = Try(spark.read.parquet(modelPath + META_DIR).as[HSMeta].head(1).headOption) match {
      case Success(Some(elem)) => elem
      case Success(None) => throw new ModelException(s"TrainMeta dataset is empty at path ${modelPath + META_DIR}")
      case Failure(e) => throw new ModelException(s"Failed to load TrainMeta from path ${modelPath + META_DIR}: ${e.getMessage}", e)
    }

    logger.info(s"Loaded training metadata: minTs=${meta.minTs}, rangeTs=${meta.rangeTs}, trainSigmas=${meta.trainSigmas}, " +
      s"nChangepoints=${meta.nChangepoints}, fourierOrders=[H=${meta.fourierOrderHourly}, W=${meta.fourierOrderWeekly}, " +
      s"M=${meta.fourierOrderMonthly}, Y=${meta.fourierOrderYearly}]")

    HSModel(model, meta)
  }

}
