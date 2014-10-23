package ml

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SchemaRDD

class Dataset(val rdd: SchemaRDD) {

  def project(cols: Seq[String]) = {
    // sql("SELECT " + input + " from " + dataset.name)
    rdd
  }

  def append(other: SchemaRDD): SchemaRDD = {
    // rdd.zip(other)
    rdd
  }

}
