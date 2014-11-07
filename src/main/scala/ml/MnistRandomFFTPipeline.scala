package ml

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row
import org.apache.spark.sql.SQLContext

object MnistRandomFFTPipeline {

  def main(args: Array[String]) {
    if (args.length != 4) {
      println("Usage: MnistRandomFFTPipeline <master> <trainingFile> <testFile> <numFFTs>")
      System.exit(0)
    }

    val startTime = System.currentTimeMillis

    val sparkMaster = args(0)
    val trainingFile = args(1)
    val testFile = args(2)
    val numFFTs = args(3).toInt

    val numParts = 10
    val numClasses = 10

    val conf = new SparkConf().setMaster(sparkMaster)
      .setAppName("MnistRandomFFTPipeline")
      .set("spark.executor.memory", "2g")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    import sqlContext._

    val seed = 0L
    val random = new java.util.Random(seed)
    val randomSignSource = new java.util.Random(random.nextLong())

    val d = 784

    val train = sc.textFile(trainingFile, numParts).map { x =>
      val parts = x.split(",").map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.drop(1)))
    }

    val test = sc.textFile(testFile, numParts).map { x =>
      val parts = x.split(",").map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.drop(1)))
    }

    val convertLabels = 
      new ClassLabelIndicators(numClasses).setInputCol("label").setOutputCol("labelClasses")

    val randomSignNode = RandomSignNode.create(d, randomSignSource)
      .setInputCol("features")
      .setOutputCol("randomFeatures")

    val fftTransform = new FFTransform()
      .setInputCol("randomFeatures")
      .setOutputCol("fftFeatures")

    val softMax = new LinearRectifier()
      .setMaxVal(0.0)
      .setInputCol("fftFeatures")
      .setOutputCol("softMaxFeatures")

    val linearSolver = 
      new MultiClassLinearRegressionEstimator()
        .setFeaturesCol("softMaxFeatures")
        .setLabelCol("labelClasses")
        .setOutputCol("predictedLabels")

    // TODO: How do we do multiple batches of Random+FFT+Max before LinearRegression ?
    val pipeline = new Pipeline().setStages(Array(
      convertLabels,
      randomSignNode,
      fftTransform,
      softMax,
      linearSolver
    ))

    val model = pipeline.fit(train)

    val predictions = model.transform(test)

    // TODO: How do I chain an evaluator to a Pipeline ?
    val err = new MaxClassifier().setLabelCol("label").setInputCol("predictedLabels").evaluate(predictions, new ParamMap())

    val endTime = System.currentTimeMillis
    println("Pipeline took " + (endTime - startTime)/1000 + " s" + " Error was " + err)

    sc.stop()
    System.exit(0)
  }
}
