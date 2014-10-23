package ml

import org.apache.spark.rdd.RDD
import ml._

class FeatureHasher(
  var input: String,
  var output: String,
  var numFeatures: Int) extends Transformer {

  def this() = {
    this("", "", 0)
  }

  def transform(dataset: Dataset): Dataset = {
    dataset
  }

  def transform(in: RDD[Array[_]]): RDD[Array[Double]] = {
    in.map { features =>
      val hashedArr = new Array[Double](numFeatures)
      features.map { f =>
        hashedArr(f.hashCode % numFeatures) += 1
      }
      hashedArr
    }
  }
}
