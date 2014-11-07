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

/**
 * Linear regression (example).
 */
class MultiClassLinearRegressionEstimator extends Estimator[MultiClassLinearModel]
    with HasLabelCol with HasFeaturesCol with HasOutputCol {

  // Overwrite the return type of setters for Java users.
  override def setLabelCol(labelCol: String): this.type = super.setLabelCol(labelCol)
  override def setFeaturesCol(featuresCol: String): this.type = super.setFeaturesCol(featuresCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  override def fit(dataset: SchemaRDD, paramMap: ParamMap): MultiClassLinearModel = {
    import dataset.sqlContext._
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    val A = dataset.select(this.paramMap(featuresCol).attr).map { 
      case Row(v: Vector) => v
    }.mapPartitions { part =>
      val rows = new ArrayBuffer[Array[Double]]
      while (part.hasNext) {
        rows += part.next.toArray
      }
      
      val mat = BDM(rows:_*)
      Iterator.single(mat)
    }.cache()

    val b = dataset.select(this.paramMap(labelCol).attr).map {
      case Row(v: Vector) => v
    }.mapPartitions { part =>
      val rows = new ArrayBuffer[Array[Double]]
      while (part.hasNext) {
        rows += part.next.toArray
      }
      
      val mat = BDM(rows:_*)
      Iterator.single(mat)
    }.cache()
  
    val AtA = A.mapPartitions { part =>
      val a = part.next
      Iterator.single(a.t * a)
    }.treeReduce(_ + _)

    val Atb = A.zip(b).mapPartitions { part =>
      val ab = part.next
      Iterator.single(ab._1.t * ab._2)
    }.treeReduce(_ + _)

    val x = AtA \ Atb

    val lrm = new MultiClassLinearModel(Matrices.dense(x.rows, x.cols, x.data))
              .setFeaturesCol(this.paramMap(featuresCol))
              .setOutputCol(this.paramMap(outputCol))

    A.unpersist()
    b.unpersist()
    lrm
  }
}

class MultiClassLinearModel(
    val weights: Matrix)
    extends Model with HasFeaturesCol with HasOutputCol {

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    import dataset.sqlContext._
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    val predict: Vector => Vector = (v) => {
      val vec = new BDV(v.toArray)
      val mat = new BDM[Double](weights.numRows, weights.numCols, weights.toArray)
      Vectors.dense((vec.t * mat).t.data)
    }

    dataset.select(
      Star(None),
      predict.call((this.paramMap(featuresCol).attr)) as Symbol(this.paramMap(outputCol))
    )
  }
}
