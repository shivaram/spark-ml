package ml.example

import ml._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext

object SimplePipeline {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setMaster("local")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    val dataset = new Dataset(sqlContext.sql(
      """
      SELECT words from webcrawl_data;
      """
    ))

    val testDataset = new Dataset(sqlContext.sql(
      """
      SELECT words from test_data;
      """
    ))

    val featureHasher = new FeatureHasher(input="words", output="hashed", numFeatures=100)
    val lr = new LogisticRegression(maxIter=100, regParam=0.1)

    val pipeline = new Pipeline().andThen(featureHasher).andThen(lr)
    val model = pipeline.fit(dataset)

    val testPipeline = new Pipeline().andThen(featureHasher).andThen(model)
    
    val err = testPipeline.fit(testDataset)
    println("Pipeline trained with err " + err)
  }
}

