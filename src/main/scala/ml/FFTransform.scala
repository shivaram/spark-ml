package ml

import org.opencv.core._
import org.opencv.imgproc._

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

class FFTransform extends Transformer with HasInputCol with HasOutputCol {

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  /**
   * Converts FFT output from OpenCV into a DoubleMatrix.
   * Assumes the OpenCV Mat is Conjugate complex symmetric, so output contains
   * [0,1,2,3...n/2 , n/2-1,n/2-2... 1]
   */
  def fftMatToDoubleMatrix(in: Mat, temp: Array[Double]): Array[Double] = {
    in.rowRange(0, in.rows()/2 + 1).get(0, 0, temp)
    temp
  }

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    import dataset.sqlContext._
    // val map = this.paramMap ++ paramMap
    import this.paramMap.implicitMapping

    // TODO: Why can't this UDF be a function defined in this class ? 
    // How do I test this UDF ?
    val fft: Array[Double] => Array[Double] = (in) => {
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

      val numFFTVals = FFTransform.nextPowerOfTwo(in.length)
      // Allocate a temporary array to store
      val tmat = Mat.zeros(numFFTVals, 1, CvType.CV_64F)
      val dftout = new Mat(numFFTVals, 1, CvType.CV_64FC2)

      val temp: Array[Double] = Array.fill(numFFTVals/2)(0.0)

      tmat.put(0, 0, in.map(x => x.toDouble):_*)

      val li = new java.util.ArrayList[Mat]

      Core.dft(tmat, dftout, Core.DFT_COMPLEX_OUTPUT, 0)
      Core.split(dftout, li)

      val fft = fftMatToDoubleMatrix(li.get(0), temp).clone()

      li.get(0).release()
      li.get(1).release()
      li.clear()

      assert(fft.forall(!_.isNaN))
      tmat.setTo(new Scalar(0.0))

      fft
    }

    // TODO: Need to call mapPartitions !
    // TODO: Or we need an easier way to put an RDD into the schemaRDD ?
    dataset.select(Star(None), fft.call(this.paramMap(inputCol).attr) as
      Symbol(this.paramMap(outputCol)))
  }
}

object FFTransform extends Serializable {
  def nextPowerOfTwo(numInputFeatures: Int) = {
    math.pow(2, math.ceil(math.log(numInputFeatures)/math.log(2))).toInt
  }
}
