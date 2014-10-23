package ml

abstract class Estimator extends PipelineStage {
  def fit(dataset: Dataset): Transformer
}
