//Importanciones necesarias para el metodo de Machine Learning y Limpieza de datos
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}

/* Del Data set Iris.csv que se encuentra en el repositorio del prof elaborar la limpieza de datos 
necesaria para ser procesado por el siguiente algotmo (limpieza por medio de scala)

Ejemplo de datos de iris.csv
5.1,      3.5,    1.4,      0.2,    setosa
Numero1, Numero2, Numero3, Numero4,  Tipo
*/
val structtype = StructType ( StructField ("Numero1", DoubleType,true)
:: StructField("Numero2", DoubleType, true)
:: StructField("Numero3", DoubleType, true)
:: StructField("Numero4", DoubleType, true)
:: StructField("Tipo", StringType, true)
:: Nil)
val dfstruct = spark.read.option("header","true").schema(structtype)csv("iris.csv")
val label = new StringIndexer().setInputCol("Tipo").setOutputCol("label")
val assembler = new VectorAssembler().setInputCols(Array("Numero1","Numero2","Numero3","Numero4")).setOutputCol("features")

/*De la libreria Mlib de Spark utilice el algoritmo de machine Learning llamado MultiPlayer Perceptrion */
//60% de entrenamiento
//40% test
val splits = dfstruct.randomSplit(Array(0.6,0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

/*Diseniar su propia arquitectura con un minimo de tres neuronas de capa de entrada, dos o mas 
neuronas en la capa oculta y tres neuronas en la capa de salida */
//4 neuronas en la capa de entrada
//2 capas ocultas una de 5 neuronas y otra de 4
//3 neuronas en la capa de sallida
val layers = Array[Int](4, 5, 4, 3)

/*Explique detalladamente la funcion matematica de entrenamiento que utiliza estte algoritmo por defecto
Crea un entrenamiento en base a las conecciones entre neuronas, les da peso a las mismas entre un rango de 0 y 1
*/
 // Entrenamiento del modelo
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
val pipe = new Pipeline().setStages(Array(label,assembler,trainer))
val model = pipe.fit(train)


/*Explique con sus propias palabras la funcion matematica del error que utiliza este algoritmo
compara los de entretamiento con los de test, y saca en error con una funcion predeterminada  */
val res = model.transform(test)
res.show()
val predictionAndLabels = res.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

/* Finalmente suba su codigo a la rama en su github y documente en el readme como en google drive */




