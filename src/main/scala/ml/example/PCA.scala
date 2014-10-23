package ml.example

import ml._

class PCA(override val id: String, var k: Int) extends Estimator {

  def this() = this("pca-" + Identifiable.randomId(), 10)

  override def fit(dataset: Dataset): PCA.Model = null
}

object PCA {

  class Model(override val id: String) extends Transformer {
    override def transform(dataset: Dataset): Dataset = null
  }
}
