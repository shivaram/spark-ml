package ml

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors, Matrices, Matrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

import org.apache.spark.mllib.rdd.RDDFunctions._

import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV,
  squaredDistance => breezeSquaredDistance, DenseMatrix => BDM}

class MaxClassifier extends Evaluator with Params with HasInputCol with HasLabelCol {

  // TODO: Label here represents actual?
  override def setLabelCol(labelCol: String): this.type = super.setLabelCol(labelCol)
  // TODO: Input here represents computed labels ?
  // TODO: Need better naming 
  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)

  def computeErr(actual: Double, computed: Vector) = {
    val arr = computed.toArray
    val index = arr.indexOf(arr.max)
    if (index.toDouble != (actual - 1.0)) {
      1.0
    } else {
      0.0
    }
  }

  override def evaluate(dataset: SchemaRDD, paramMap: ParamMap): Double = {
    import dataset.sqlContext._
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    val err = dataset.select(this.paramMap(labelCol).attr, this.paramMap(inputCol).attr).map {
        case r: Row =>
          val actual = r(0).asInstanceOf[Double]
          val computed = r(1).asInstanceOf[Vector]
          computeErr(actual, computed)
    }.reduce(_ + _)

    val total = dataset.count

    err / total * 100.0
  }

}
