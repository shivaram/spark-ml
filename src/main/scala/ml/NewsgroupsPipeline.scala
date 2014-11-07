package ml

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.test.TestSQLContext._

case class Article(label: Int, doc: String)
object NewsgroupsPipeline {

  def main(args: Array[String]) {
    if (args.length != 4) {
      println("Usage: MnistRandomFFTPipeline <master> <trainingFile> <testFile> <numFFTs>")
      System.exit(0)
    }

    val startTime = System.currentTimeMillis

    val sparkMaster = args(0)
    val conf = new SparkConf().setMaster(sparkMaster)
      .setAppName("MnistRandomFFTPipeline")
      .set("spark.executor.memory", "2g")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    val numClasses = 20

    val trainData = List(
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/alt.atheism").map(0 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/comp.graphics").map(1 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/comp.os.ms-windows.misc").map(2 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/comp.sys.ibm.pc.hardware").map(3 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/comp.sys.mac.hardware").map(4 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/comp.windows.x").map(5 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/misc.forsale").map(6 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/rec.autos").map(7 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/rec.motorcycles").map(8 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/rec.sport.baseball").map(9 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/rec.sport.hockey").map(10 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/sci.crypt").map(11 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/sci.electronics").map(12 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/sci.med").map(13 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/sci.space").map(14 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/soc.religion.christian").map(15 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/talk.politics.guns").map(16 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/talk.politics.mideast").map(17 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/talk.politics.misc").map(18 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train/talk.religion.misc").map(19 -> _._2)
    ).reduceLeft(_.union(_))

    val testData = List(
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/alt.atheism").map(0 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/comp.graphics").map(1 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/comp.os.ms-windows.misc").map(2 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/comp.sys.ibm.pc.hardware").map(3 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/comp.sys.mac.hardware").map(4 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/comp.windows.x").map(5 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/misc.forsale").map(6 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/rec.autos").map(7 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/rec.motorcycles").map(8 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/rec.sport.baseball").map(9 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/rec.sport.hockey").map(10 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/sci.crypt").map(11 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/sci.electronics").map(12 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/sci.med").map(13 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/sci.space").map(14 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/soc.religion.christian").map(15 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/talk.politics.guns").map(16 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/talk.politics.mideast").map(17 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/talk.politics.misc").map(18 -> _._2),
      sc.wholeTextFiles("/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test/talk.religion.misc").map(19 -> _._2)
    ).reduceLeft(_.union(_))

    val dataset = trainData.map(x => Article(x._1, x._2))
    val tokenizer: SimpleNGramTokenizer = new SimpleNGramTokenizer().setInputCol("doc").setOutputCol("tokens")
    val featureSelector = new MostFrequentSparseFeatureSelector(100000).setInputCol("tokens")
    val bayesModel = new NaiveBayesEstimator().setLabelCol("label").setFeaturesCol("features")
    val pipeline: Pipeline = new Pipeline().setStages(Array[PipelineStage](tokenizer, featureSelector, bayesModel))
    val model: PipelineModel = pipeline.fit(createSchemaRDD(dataset))

    // Evaluate the classifier
    println("Evaluating classifier")
    //val evaluator = (x: RDD[(Int, String)]) => MulticlassClassifierEvaluator.evaluate(x.mapPartitions(predictionPipeline.transform))
    //logger.info("Train Dataset:\n" + evaluator(trainData).mkString("\n"))
    //println("Test Dataset:\n" + evaluator(testData).mkString("\n"))

    sc.stop()
    System.exit(0)
  }
}
