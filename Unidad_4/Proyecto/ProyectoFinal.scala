//Contenido del proyecto
//1.- Objectivo: Comparacion del rendimiento siguientes algoritmos de machine learning
// - SVM
// - Decision Tree
// - Logistic Regresion
// - Multilayer perceptron
//Con el siguiente data set: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing 

// Contenido del documento de proyecto final
// 1. Portada
// 2. Indice
// 3. Introduccion
// 4. Marco teorico de los algoritmos
// 5. Implementacion (Que herramientas usaron y porque (en este caso spark con scala))
// 6. Resultados (Un tabular con los datos por cada algoritmo para ver su preformance)
//    y su respectiva explicacion.
// 7. Conclusiones
// 8. Referencias (No wikipedia por ningun motivo, traten que sean de articulos cientificos)
//    El documento debe estar referenciado 

// Nota: si el documento no es presentado , no revisare su desarrollo del proyecto

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder, IndexToString}
import org.apache.log4j._
import org.apache.spark.mllib.evaluation.MulticlassMetrics

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

data.printSchema()
data.head(1)

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","balance","day","duration","campaign","pdays","previous"))
                  .setOutputCol("features"))


// - SVM

// - Decision Tree
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, dt))
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("label", "features").show(5)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)

// - Logistic Regresion
val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,lr))
val model = pipeline.fit(training)
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy

// - Multilayer perceptron

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed=1234L)
val capas = Array[Int](7, 5, 5, 2)
val mlp = new MultilayerPerceptronClassifier().setLayers(capas).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,mlp))
val modelo = pipeline.fit(trainingData)
val resultado = modelo.transform(testData)
val predictionAndLabels = resultado.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))