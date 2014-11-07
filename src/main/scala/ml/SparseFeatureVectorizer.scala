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

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{ParamMap, HasOutputCol, HasInputCol, Params}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.expressions.Row


/**
 * A transformer which given a feature space, maps extracted features
 * of the form (name, value) into a sparse vector
 */
class SparseFeatureVectorizer(featureSpace: Map[String, Int])
    extends Model with Params with HasInputCol with HasOutputCol {

  override def setInputCol(inputCol: String): this.type = super.setInputCol(inputCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)

  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    val map = this.paramMap ++ paramMap
    val input = dataset.select((inputCol: String).attr)
        .map { case Row(v: Seq[(String, Double)]) =>
      v
    }

    val featurize: (Seq[(String, Double)]) => Vector = (in) => {
      val features = in.map(f => (featureSpace.get(f._1), f._2))
          .filter(_._1.isDefined)
          .map(f => (f._1.get, f._2))
      Vectors.sparse(featureSpace.size, features)
    }

    dataset.select(Star(None), featurize.call((inputCol: String).attr) as outputCol)
  }
}
