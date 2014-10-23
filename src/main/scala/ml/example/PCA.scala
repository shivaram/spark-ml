package ml.example

import ml._

class PCA(var k: Int) extends Estimator {

  def this() = this(10)

  override def fit(dataset: Dataset): PCA.Model = null
}

object PCA {

  class Model extends Transformer {
    override def transform(dataset: Dataset): Dataset = null
  }
}
