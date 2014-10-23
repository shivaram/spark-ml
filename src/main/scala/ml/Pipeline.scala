package ml

import scala.collection.mutable.ListBuffer

trait PipelineStage extends Identifiable

class Pipeline(override val id: String, var stages: Seq[PipelineStage]) extends Estimator {

  def this() = this("Pipeline-" + Identifiable.randomId(), Seq[PipelineStage]())

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

  class Model(override val id: String, val transformers: Array[Transformer]) extends Transformer {

    def this(transformers: Array[Transformer]) = this("Pipeline.Model-" + Identifiable.randomId(), transformers)

    override def transform(dataset: Dataset): Dataset = {
      transformers.foldLeft(dataset) { (dataset, transformer) =>
        transformer.transform(dataset)
      }
    }
  }
}

