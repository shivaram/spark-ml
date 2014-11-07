package ml

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

class LinearRectifier extends Transformer with HasInputCol with HasOutputCol {

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  val maxVal: DoubleParam = new DoubleParam(this, "regParam", "maximum value for soft max",
    Some(0.0))
  def setMaxVal(maxVal: Double): this.type = {
    set(this.maxVal, maxVal)
    this
  }
  
  def getMaxVal: Double = {
    get(maxVal)
  }

  val alpha: DoubleParam = new DoubleParam(this, "alpha", "linear shift value", Some(0.0))
  def setAlpha(alpha: Double): this.type = {
    set(this.alpha, alpha)
    this
  }
  
  def getAlpha: Double = {
    get(alpha)
  }

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    import dataset.sqlContext._
    import this.paramMap.implicitMapping

    val softMax: Vector => Vector = (in) => {
      Vectors.dense(in.toArray.map(e => math.max(this.paramMap(maxVal), e - this.paramMap(alpha))))
    }

    dataset.select(Star(None), softMax.call(this.paramMap(inputCol).attr) as
      Symbol(this.paramMap(outputCol)))
  }
}
