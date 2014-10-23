package ml

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SchemaRDD

class Dataset(val rdd: SchemaRDD) {

  def select(col: String) = {
    // sql("SELECT " + input + " from " + dataset.name)
    rdd
  }

  def bind(col: String, colRDD: RDD[_]) =  {
    rdd
  }

}
