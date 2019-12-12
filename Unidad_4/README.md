- [**Introduccion**](#Introduccion)
- [**Marco teorico de los algoritmos**](#Marco-teorico-de-los-algoritmos)
- [**Implementacion**](#Implementacion)
- [**Resultados**](#Resultados)
- [**Conclusiones**](#Conclusiones)
- [**Referencias**](#Referencias)

# Introduccion

En este proyecto analizaremos 3 algoritmos de machine learning como lo son los Decision Tree, Logistic Regression, Multilayer perceptron. De cada uno de estos algoritmos miraremos una breve introducción a cada uno para saber cual es el funcionamiento del mismo, así como la aplicación en un Data Set (conjunto de datos definido) para poder analizar los distintos valores de exactitud, así como el poder ver el comportamiento de cada uno.

# Marco teorico de los algoritmos

## Decision Tree
Árboles de decisión Un árbol de decisión es un modelo de predicción cuyo objetivo principal es el aprendizaje inductivo a partir de observaciones y construcciones lógicas. Son muy similares a los sistemas de predicción basados en reglas, que sirven para representar y categorizar una serie de condiciones que suceden de forma sucesiva para la solución de un problema. Constituyen probablemente el modelo de clasificación más utilizado y popular. El conocimiento obtenido durante el proceso de aprendizaje inductivo se representa mediante un árbol. Un árbol gráficamente se representa por un conjunto de nodos, hojas y ramas. El nodo principal o raíz es el atributo a partir del cual se inicia el proceso de clasificación; los nodos internos corresponden a cada una de las preguntas acerca del atributo en particular del problema. Cada posible respuesta a los cuestionamientos se representa mediante un nodo hijo. Las ramas que salen de cada uno de estos nodos se encuentran etiquetadas con los posibles valores del atributo [1]. Los nodos finales o nodos hoja corresponden a una decisión, la cual coincide con una de las variables clase del problema a resolver. 
Este modelo se construye a partir de la descripción narrativa de un problema, ya que provee una visión gráfica de la toma de decisión, especificando las variables que son evaluadas, las acciones que deben ser tomadas y el orden en el que la toma de decisión será efectuada. Cada vez que se ejecuta este tipo de modelo, sólo un camino será seguido dependiendo del valor actual de la variable evaluada. Los valores que pueden tomar las variables para este tipo de modelos pueden ser discretos o continuos [2]. 

Un algoritmo de generación de árboles de decisión consta de 2 etapas: la primera corresponde a la inducción del árbol y la segunda a la clasificación. En la primera etapa se construye el árbol de decisión a partir del conjunto de entrenamiento; comúnmente cada nodo interno del árbol se compone de un atributo de prueba y la porción del conjunto de entrenamiento presente en el nodo es dividida de acuerdo con los valores que pueda tomar ese atributo. La construcción del árbol inicia generando su nodo raíz, eligiendo un atributo de prueba y dividiendo el conjunto de entrenamiento en dos o más subconjuntos; para cada partición se genera un nuevo nodo y así sucesivamente. Cuando en un nodo se tienen objetos de más de una clase se genera un nodo interno; cuando contiene objetos de una clase solamente, se forma una hoja a la que se le asigna la etiqueta de la clase. En la segunda etapa del algoritmo cada objeto nuevo es clasificado por el árbol construido; después se recorre el árbol desde el nodo raíz hasta una hoja, a partir de la que se determina la membresía del objeto a alguna clase. El camino a seguir en el árbol lo determinan las decisiones tomadas en cada nodo interno, de acuerdo con el atributo de prueba presente en él.


 ## Logistic Regression
La técnica de la regresión logística se originó en la década de los 60 con el trabajo de Cornfield, Gordon y Smith [3] en 1967 Walter y Duncan la utilizan ya en la forma que la conocemos actualmente, o sea para estimar la probabilidad de ocurrencia de un proceso en función de ciertas variables. [4]. Su uso se incrementa desde principios de los 80 como consecuencia de los adelantos ocurridos en el campo de la computación.

El objetivo de esta técnica estadística es expresar la probabilidad de que ocurra un hecho como función de ciertas variables, supongamos que son k (k ³ 1), que se consideran potencialmente influyentes. La regresión logística, al igual que otras técnicas estadísticas multivariadas, da la posibilidad de evaluar la influencia de cada una de las variables independientes sobre la variable respuesta y controlar el efecto del resto. Tendremos, por tanto, una variable dependiente, llamémosla Y, que puede ser dicotómica o politómica (en este trabajo nos referiremos solamente al primer caso) y una o más variables independientes, llamémosle X.

Al ser la variable Y dicotómica, podrá tomar el valor "O" si el hecho no ocurre y "1" si el hecho ocurre; el asignar los valores de esta manera o a la inversa es intrascendente, pero es muy importante tener en cuenta la forma en que se ha hecho llegado el momento de interpretar los resultados. Las variables independientes (también llamadas explicativas) pueden ser de cualquier naturaleza: cualitativas o cuantitativas. La probabilidad de que Y=1 se denotará por p.

La forma analítica en que la probabilidad objeto de interés se vincula con las variables explicativas es la siguiente. [5]

Esta expresión es la que se conoce como función logística; donde exp denota la función exponencial y a1, b1, b2... bk son los parámetros del modelo. Al producir la función exponencial valores mayores que 0 para cualquier argumento, p tomará sólo valores entre 0 y 1.

Si b es positiva (mayor que 0) entonces la función es creciente y decreciente en el caso contrario. Un coeficiente positivo indica que p crece cuando lo hace la variable.


## Multilayer perceptron
Las RNA de tipo Perceptrón Multicapa (PM) se encuentran entre las arquitecturas de red más poderosas y populares. Están formadas por una capa de entrada, un número arbitrario de capas ocultas, y una capa de salida. Cada una de las neuronas ocultas o de salida recibe una entrada de las neuronas de la capa previa (conexiones hacia atrás), pero no existen conexiones laterales entre las neuronas dentro de cada capa [6]

La capa de entrada contiene tantas neuronas como categorías correspondan a las variables independientes que se desean representar. La capa de salida corresponde a la variable respuesta, que en este caso es una variable categórica. 


# Implementacion

```Scala
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

//Preparacion de datos
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","balance","day","duration","campaign","pdays","previous"))
                  .setOutputCol("features"))


// - Decision Tree
val Array(trainingData, testData) = data.randomSplit(Array(0.6, 0.4))
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, dt))
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.select("label", "features").show(5)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Testo Error = " + (1.0 -accuracy))
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)
println("////////////////////////////////////////////////////////////////////////")

// - Logistic Regresion
val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,lr))
val model = pipeline.fit(training)
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy Logistic Regresion = " + metrics.accuracy)
println("////////////////////////////////////////////////////////////////////////")

// - Multilayer perceptron

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed=1234L)
val capas = Array[Int](7,5,3,2)
val mlp = new MultilayerPerceptronClassifier().setLayers(capas).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,mlp))
val modelo = pipeline.fit(trainingData)
val resultado = modelo.transform(testData)
val predictionAndLabels = resultado.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println("Test set accuracy Multilayer perceptron = " + evaluator.evaluate(predictionAndLabels))
```

# Resultados

#### Tabla de Exactitud

| Algoritmo              |    Exactitud Maxima   |
|------------------------|-----------------------|
| Decision Tree          | 0.890240328253224     |
| Logistic Regression    | 0.8910135233386651    |
| Multilayer perceptron  | 0.8827505142521305    | 

 Tabla de Decision Tree

| Porcentaje de datos    |      Medidas de                    |
| de prueba y test       |    tiempo y memoria                |
|------------------------|------------------------------------|
| Prueba 60% Test 40%    | Used Memory: 250.84401599999998    |
|                        | durationSeconds: Long = 25         |
| Prueba 70% Test 30%    | Used Memory: 401.83395199999995    |
|                        | durationSeconds: Long = 16         |
| Prueba 80% Test 20%    | Used Memory: 330.96304             |
|                        | durationSeconds: Long = 36         |
| Prueba 90% Test 10%    | Used Memory: 460.322856            |
|                        | durationSeconds: Long = 22         |
| Prueba 50% Test 50%    | Used Memory: 258.89707999999996    |
|                        | durationSeconds: Long = 17         |

Tabla de Multilayer perceptron

| Porcentaje de datos    |      Medidas de                    |
| de prueba y test       |    tiempo y memoria                |
|------------------------|------------------------------------|
| Prueba 60% Test 40%    | Used Memory: 299.515384            |
|          (7,5,3,2)     | durationSeconds: Long =  41        |
| Prueba 70% Test 30%    | Used Memory: 431.11094399999996    |
|        (7,4,3,2)       | durationSeconds: Long = 24         |
| Prueba 60% Test 40%    | Used Memory: 294.59689599999996    |
|      (7,4,2,2)         | durationSeconds: Long = 22         |
| Prueba 50% Test 50%    | Used Memory: 300.34875999999997    |
|      (7,5,3,2)         | durationSeconds: Long = 51         |
| Prueba 90% Test 10%    | Used Memory: 471.933632            |
|     (7,5,2,2)          | durationSeconds: Long = 32         |

Tabla de Logistic Regresion

| Porcentaje de datos    |      Medidas de                    |
| de prueba y test       |    tiempo y memoria                |
|------------------------|------------------------------------|
| Prueba 60% Test 40%    | Used Memory: 370.954752            |
|                        | durationSeconds: Long = 38         |
| Prueba 70% Test 30%    | Used Memory: 325.091               |
|                        | durationSeconds: Long = 15         |
| Prueba 80% Test 20%    | Used Memory: 429.23424             |
|                        | durationSeconds: Long = 25         |
| Prueba 90% Test 10%    | Used Memory: 368.6954              |
|                        | durationSeconds: Long = 46         |
| Prueba 50% Test 50%    | Used Memory: 405.25988             |
|                        | durationSeconds: Long = 19         |


# Conclusiones

En este proyecto se puede analizar como a un mismo conjunto de datos se le pueden aplicar varias técnicas de algoritmos de machine learning, y cada una de estas da un resultado.
Es bueno en analizar el resultado de los diferentes algoritmos con el afán de encontrar el mejor.
Un de talle un poco fuera del algoritmo, pude notar que spark con Scala son un poco más rápido y preciso.

En cuestion de tiempo la regresion logica es mas rapida, mas la exactitud es la mas baja de las 3, y en general es la que mas memoria usa.

Multilayer perceptron es la que mas tiempo duro en general, mas su exactitud es buena y el uso de memoria es similar al de arboles.

# Referencias
[1] Russell, S. and P. Norvig, Artificial Intelligence: A Modern Approach. Second ed. Upper Saddle River (N J): Prentice Hall/ Pearson Education; 2003. 
[2] Breiman L, Friedman JH, Olshen RA, Stone CJ. Classification and Regression Trees, Wadsworth (New York); 1994
[3] Cornfield J, Gordon T, Smith WN. Quantal response curves for experimentally uncontroled variables. Bull Int Statist Inst 1961;38:97-115.
[4] Walter S, Duncan D. Estimation of the probability of an event as a function of several variables. Biometrika 1967;54:167-79.
[5] Silva LC. Excursión a la regresión logística en ciencias de la salud. Madrid:Díaz Santos, 1994:3-11.
[6] Castillo, E., Cobo, A., Gutiérrez, J.M., Pruneda, R.E.: Introducción a las Redes Funcionales con Aplicaciones. Un Nuevo Paradigma Neuronal. Editorial Paraninfo S.A. Madrid. España. pp.5-8; 8-16; 21-24, 30-34, 53-100. (1999) 
