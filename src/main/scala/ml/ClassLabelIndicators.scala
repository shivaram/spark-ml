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

class ClassLabelIndicators(numClasses: Int) extends Transformer with Params with HasInputCol with HasOutputCol { 

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    import dataset.sqlContext._
    // FIXME: I get an error saying ++ is private !
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    val createClassLabels: Double => Vector = (in) => {
      val a = Array.fill(numClasses)(-1.0)
      a(in.toInt - 1) = 1.0
      Vectors.dense(a)
    }

    dataset.select(Star(None),
      createClassLabels.call(this.paramMap(inputCol).attr) as Symbol(this.paramMap(outputCol)))
  }
}
