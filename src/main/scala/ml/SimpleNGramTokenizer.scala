/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

/**
 * Created by tomerk11 on 10/26/14.
 */
class SimpleNGramTokenizer extends Transformer with Params with HasInputCol with HasOutputCol {

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    val map = this.paramMap ++ paramMap
    val input = dataset.select((inputCol: String).attr)
        .map { case Row(v: String) =>
      v
    }.cache()

    val tokenize: (String) => Seq[(String, Double)] = (text) => {
      getNgrams(text).distinct.map((_, 1d))
    }

    dataset.select(Star(None), tokenize.call((inputCol: String).attr) as Symbol(outputCol))
  }

  val ns = Array(1,2,3)

  def getNgrams(text: String): Seq[String] = {
    val unigrams = text.trim.toLowerCase.split("[\\p{Punct}\\s]+")
    ns.map(n => {
      unigrams.sliding(n).map(gram => gram.mkString(" "))
    }).flatMap(identity)
  }
}
