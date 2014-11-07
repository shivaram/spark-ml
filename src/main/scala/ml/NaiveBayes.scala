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

import org.apache.spark.mllib.classification.{NaiveBayes => nb}

/**
 * Logistic regression (example).
 */
class NaiveBayesEstimator extends Estimator[NaiveBayesTransformer]
    with HasLabelCol with HasFeaturesCol {

  // Overwrite the return type of setters for Java users.
  override def setLabelCol(labelCol: String): this.type = super.setLabelCol(labelCol)
  override def setFeaturesCol(featuresCol: String): this.type = super.setFeaturesCol(featuresCol)

  override final val modelParams: NaiveBayesModelParams = new NaiveBayesModelParams {}

  override def fit(dataset: SchemaRDD, paramMap: ParamMap): NaiveBayesTransformer = {
    val map = this.paramMap ++ paramMap
    val instances = dataset.select((labelCol: String).attr, (featuresCol: String).attr)
      .map { case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
      }

    val model = new NaiveBayesTransformer(nb.train(instances))

    this.modelParams.params.foreach { param =>
      if (map.contains(param)) {
        model.paramMap.put(model.getParam(param.name), map(param))
      }
    }
    if (!model.paramMap.contains(model.featuresCol) && map.contains(model.featuresCol)) {
      model.setFeaturesCol(featuresCol)
    }
    model
  }
}

trait NaiveBayesModelParams extends Params with HasFeaturesCol with HasOutputCol {
  override def setFeaturesCol(featuresCol: String): this.type = super.setFeaturesCol(featuresCol)
  override def setOutputCol(outputCol: String): this.type = super.setOutputCol(outputCol)
}

class NaiveBayesTransformer (
    val model: NaiveBayesModel)
    extends Model with NaiveBayesModelParams {


  override def transform(dataset: SchemaRDD, paramMap: ParamMap): SchemaRDD = {
    val map = this.paramMap ++ paramMap
    println(s"transform called with $map")

    val predict: Vector => Double = (v) => {
      model.predict(v)
    }
    dataset.select(
      Star(None),
      predict.call((featuresCol: String).attr) as outputCol)
  }
}
