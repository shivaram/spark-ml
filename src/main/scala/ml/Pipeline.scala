package ml

import scala.collection.mutable.ListBuffer

trait PipelineStage extends Identifiable

class Pipeline(initStages: PipelineStage*) extends Estimator {

  val stages = new ListBuffer[PipelineStage]
  stages.appendAll(initStages)

  def andThen(stage: PipelineStage): Pipeline = {
    stages += stage
    this
  }

  override def fit(dataset: Dataset): Transformer = {
    // Search for last estimator.
    var lastIndexOfEstimator = -1
    stages.view.zipWithIndex.foreach { case (stage, index) =>
      stage match {
        case _: Estimator =>
          lastIndexOfEstimator = index
        case _ =>
      }
    }
    var curDataset = dataset
    val transformers = ListBuffer.empty[Transformer]
    stages.view.zipWithIndex.foreach { case (stage, index) =>
      stage match {
        case estimator: Estimator =>
          val transformer = estimator.fit(dataset)
          if (index < lastIndexOfEstimator) {
            curDataset = transformer.transform(curDataset)
          }
          transformers += transformer
        case transformer: Transformer =>
          if (index < lastIndexOfEstimator) {
            curDataset = transformer.transform(curDataset)
          }
          transformers += transformer
        case _ =>
          throw new IllegalArgumentException
      }
    }

    new Pipeline.Model(transformers.toArray)
  }
}

object Pipeline {

  class Model(val transformers: Array[Transformer]) extends Transformer {

    override def transform(dataset: Dataset): Dataset = {
      transformers.foldLeft(dataset) { (dataset, transformer) =>
        transformer.transform(dataset)
      }
    }
  }
}

