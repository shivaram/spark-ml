package ml.example

import ml._

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row

class LogisticRegression(
  override val id: String, 
  var maxIter: Int, 
  var regParam: Double) extends Estimator {

  def this() = this("lr-" + Identifiable.randomId(), 100, 0.1)

  // val maxIter: Param[Int] = new Param(this, "maxIter", "max number of iterations", Some(100))
  // 
  // val regParam: Param[Double] = new Param(this, "regParam", "regularization constant", Some(0.1))

  override def fit(dataset: Dataset): LogisticRegression.Model = {
    // val sqlContext = dataset.sqlContext
    // import sqlContext._
    // val instances = dataset.select('label, 'features).map { case Row(label: Double, features: Vector) =>
    //   LabeledPoint(label, features)
    // }.cache()
    // val lr = new LogisticRegressionWithLBFGS
    // lr.optimizer
    //   .setRegParam(regParam)
    //   .setNumIterations(maxIter)
    // val model = lr.run(instances)
    // new LogisticRegression.Model(id + ".model", model.weights)
    new LogisticRegression.Model(id + ".model", Vectors.zeros(10))
  }
}

object LogisticRegression {
  class Model(override val id: String, weight: Vector) extends Transformer {
    override def transform(dataset: Dataset): Dataset = {
      null
    }
  }
}
