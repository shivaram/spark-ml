val dataset = sqlContext.sql("""
  SELECT impressionId, impression.userId, impression.adId, click.clickId,
    impressionTime, browser, ip, userFeatures, adFeatures, clickTime IS NOT NULL AS label
  FROM impression
    LEFT OUTER JOIN click ON impresssion.impressionId = click.impressionId
    JOIN user ON impression.userId = user.userId
    JOIN ad ON impression.adId = ad.adId;""")

val stratifiedSampler = new StratifiedSampler(
  stratum="label", fractions={True: 1.0, False: 0.01, None: 1.0}, seed=11)

val timeTransformer = new TimeTransformer(
  input="impressionTime", dataType=TimeTransformer.HOUR_OF_DAY, output="impressionHour")

val indexer = new Indexer(input=Seq("browser", "country"), orderByFreq=true)

val ipLocator = new IpLocator()

val interactor = new FeatureInteractor(
  input=Seq("userFeatures$gender", "adFeatures$targetGender"), output="genderMatch")

val oneHotEncoder = new OneHotEncoder(input=Seq("countries", "browsers"),
  output=Seq("countryIndex", "browserIndex"), deactiveInputFeatures=true)

val fvAssembler = new FeatureVectorAssembler(output="features",
  input=Seq("userFeatures", "adFeatures", "impressionHour", "countries", "browsers", "genderMatch"))

val evaluator = new BinaryClassificationEvaluator(metric=Evaluator.AREA_UNDER_ROC)
val lr = new LogisticRegression(maxIter=50, regParam=0.1, regType=Regularizers.L2)

val cv = new CrossValidator(nFold=3, estimator=lr, evaluator=evaluator)
val ir = new IsotonicRegresion(input="prediction", output="ctr")

val pipeline = new Pipeline(Seq(stratifiedSampler, timeTransformer, indexer, ipLocator,
  interactor, oneHotEncoder, fvAssembler, cv, ir))

val model = pipeline.fit(dataset)

// Change iterations and run again. Note that LR should be the same reference, so we can modify
// it here and run again
lr.maxIter = 100

val model1 = pipeline.fit(dataset)

// Get the StratifiedSampler, change its param and run again
// Default name should be the class name ?
// Or it'll be cool to have pipeline.get('stratifiedSampler).seed = 100 ?
pipeline.get("StratifiedSampler").seed = 100

val model2 = pipeline.fit(dataset)

// TODO: Hyperparameter tuning
