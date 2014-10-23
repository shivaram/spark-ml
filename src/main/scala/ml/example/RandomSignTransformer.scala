package ml.example

import scala.util.Random

import org.apache.spark.rdd.RDD

import ml._

class RandomSignTransformer(
  override val id: String,
  var input: String,
  var output: String) extends Transformer {

  // Empty constructor for use from Python.
  def this() = {
    this("random-sign-" + Identifiable.randomId(), "", "")
  }

  // TODO: Can we make this implicit ?
  def transform(dataset: Dataset): Dataset = {
    // val in = dataset.select(in)
    // Should do something like

    // val in = dataset.select(in).map(x =>
    //  x.get(0).asInstanceOf[Array[Double]])

    // val out = transform(in)

    // Should append the output column 
    // dataset.bind(out, output)
    dataset
  }

  def transform(in: RDD[Array[Double]]): RDD[Array[Double]] = {
    val size = in.first.size
    val signs = new Array[Double](size)
    val random = new Random()

    for (i <- 0 until size) {
      signs(i) = if (random.nextBoolean()) 1.0 else -1.0
    }

    val signsb = in.context.broadcast(signs)
    in.map(row => hadamardProduct(row, signsb.value))
  }

  def hadamardProduct(a: Array[Double], b: Array[Double]): Array[Double] = {
    val size = a.size
    val result = new Array[Double](size)
    var i = 0
    while (i < size) {
      result(i) = a(i)*b(i)
      i += 1
    }
    result
  }
}
