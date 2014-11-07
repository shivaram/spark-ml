package ml

import java.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}

import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

import org.apache.spark.mllib.rdd._

class RandomSignNode(signs: Array[Double]) extends Transformer with Params with HasInputCol with HasOutputCol {

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    import dataset.sqlContext._
    // FIXME: I get an error saying ++ is private !
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    // TODO: Can I use a broadcast variable here ?
    val randomProduct: (Vector) => Vector = (v) => {
      Vectors.dense(RandomSignNode.hadamardProduct(v.toArray, signs))
    }

    // TODO: Need a friendler syntax for this ?
    dataset.select(Star(None),
      randomProduct.call(this.paramMap((inputCol)).attr) as Symbol(this.paramMap(outputCol)))
  }
}

object RandomSignNode {
  def create(size: Int, random: Random): RandomSignNode = {
    val signs = new Array[Double](size)
    for (i <- 0 until size) {
      signs(i) = if (random.nextBoolean()) 1.0 else -1.0
    }
    new RandomSignNode(signs)
  }

  def hadamardProduct(a: Array[Double], b: Array[Double]): Array[Double] = {
    assert(a.size == b.size, 
           "RandomSignNode: Input dimension %d does not match output dimension %d".format(
             a.size, b.size))
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
