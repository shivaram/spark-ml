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

import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors, Matrices, Matrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SchemaRDD
import org.apache.spark.sql.catalyst.analysis.Star
import org.apache.spark.sql.catalyst.dsl._
import org.apache.spark.sql.catalyst.expressions.Row

/**
 * A simple feature selector that chooses all features produced by a sparse feature extractor,
 * and produces a transformer which builds a sparse vector out of all the features it extracts
 */
class SimpleSparseFeatureSelector extends Estimator[SparseFeatureVectorizer] with HasInputCol {
  /**
   * Fits a single model to the input data with provided parameter map.
   *
   * @param dataset input dataset
   * @param paramMap parameters
   * @return fitted model
   */
  override def fit(dataset: SchemaRDD, paramMap: ParamMap): SparseFeatureVectorizer = {
    val map = this.paramMap ++ paramMap
    val instances = dataset.select((inputCol: String).attr)
        .map { case Row(features: Seq[(String, Double)]) =>
      features
    }

    val featureSpace = instances.flatMap(_.map(_._1)).distinct()
        .zipWithIndex().collect().map(x => (x._1, x._2.toInt)).toMap

    val sfv = new SparseFeatureVectorizer(featureSpace)

    this.modelParams.params.foreach { param =>
      if (map.contains(param)) {
        sfv.paramMap.put(sfv.getParam(param.name), map(param))
      }
    }
    if (!sfv.paramMap.contains(sfv.inputCol) && map.contains(sfv.inputCol)) {
      sfv.setInputCol(inputCol)
    }
    sfv
  }
}
