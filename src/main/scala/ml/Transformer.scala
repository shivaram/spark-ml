package ml

abstract class Transformer extends PipelineStage {
  def transform(dataset: Dataset): Dataset
}
