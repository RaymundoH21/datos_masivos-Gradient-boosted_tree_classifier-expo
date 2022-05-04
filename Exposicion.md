<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=650 height=150>
</p>

>**Tecnológico Nacional de México  
Instituto Tecnológico Campus Tijuana   
Carrera: Ingeniería Informática e Ingeniería en Sistemas Computacionales**
**Materia**  
Datos Masivos  
**Semestre**  
febrero- julio 2022  
**Nombre de los alumnos**  
HERNANDEZ PABLO ANAHI DEL CARMEN 
HIRALES LAZARENO RAYMUNDO
RAMOS VERDIN PAULA ANDREA
REAL CASTRO MANUEL ANGEL
**FECHA: 04/05/22**

----
# **Gradient boosted tree classifier**

## Introduction

Los árboles potenciados por gradientes (Gradient-boosted trees o GBTs) son un método de 
clasificacion y regresion basado en el uso de árboles de decisiones. El sistema de clasificación 
entrena los árboles de decisión para minimizar los fallos.

Explicacion a detalle: [AQUI](https://youtu.be/3CC4N4z3GJc?t=350)

## Decision Tree

El sistema de clasificación se encarga de crear estos árboles de decisión que permitirán obtener 
predicciones acertadas. Para que dichas predicciones sean correctas, el sistema se encarga de 
iterar los árboles prediciendo resultados para tener mejores resultados.

Explicacion a detalle: [AQUI](https://youtu.be/7VeUPuFGJHk?t=15)

## Basic Algorithm

El aumento de gradiente entrena iterativamente una secuencia de árboles de decisión. En cada 
iteración, el algoritmo usa el conjunto actual para predecir la etiqueta de cada instancia de 
entrenamiento y luego compara la predicción con la etiqueta verdadera. El conjunto de datos se 
vuelve a etiquetar para poner más énfasis en las instancias de entrenamiento con predicciones 
deficientes. Por lo tanto, en la próxima iteración, el árbol de decisiones ayudará a corregir 
los errores anteriores.

El mecanismo específico para volver a etiquetar las instancias se define mediante una función 
de pérdida (discutida a continuación). Con cada iteración, los GBT reducen aún más esta función 
de pérdida en los datos de entrenamiento.

![](https://www.researchgate.net/profile/Michael-Jahrer/publication/221654586/figure/fig1/AS:669060601769991@1536527886498/Bagged-Gradient-Boosted-Decision-Tree-A-prediction-of-the-BGBDT-consists-of-results-by-N.png)

## Data Input/Output

Una vez hecho el proceso de iteración y entrenamiento, el sistema de clasificación toma el 
Data Frame seleccionado, corre el modelo y realiza la predicción.

![](https://stackabuse.s3.amazonaws.com/media/gradient-boosting-classifiers-in-python-with-scikit-learn-3.png)

## Usage Tips

Incluimos algunas pautas para usar GBT analizando los diversos parámetros. Omitimos algunos 
parámetros del árbol de decisiones, ya que se tratan en la guía del árbol de decisiones.
- **loss:** consulte la sección anterior para obtener información sobre las pérdidas y su aplicabilidad 
a las tareas (clasificación frente a regresión). Diferentes pérdidas pueden dar resultados 
significativamente diferentes, según el conjunto de datos.
- **numIterations:** Esto establece el número de árboles en el conjunto. Cada iteración produce un 
árbol. Aumentar este número hace que el modelo sea más expresivo, lo que mejora la precisión 
de los datos de entrenamiento. Sin embargo, la precisión del tiempo de prueba puede verse 
afectada si es demasiado grande.
- **learningRate:** No es necesario ajustar este parámetro. Si el comportamiento del algoritmo parece 
inestable, disminuir este valor puede mejorar la estabilidad.
- **algo:** El algoritmo o tarea (clasificación vs. regresión) se configura mediante el parámetro 
de árbol Estrategia.

Referencias
- https://spark.apache.org/docs/latest/mllib-ensembles.html?fbclid=IwAR3bt6UeDyjVbyQZ1netRku7iIIQes0bmM_4CGDpAf_y5ZdY-GcUUXXi_F8#basic-algorithm-1
-------

### Ejemplo  
  
```scala
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a GradientBoostedTrees model.
// The defaultParams for Classification use LogLoss by default.
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Test Error = $testErr")
println(s"Learned classification GBT model:\n ${model.toDebugString}")

// Save and load model
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
val sameModel = GradientBoostedTreesModel.load(sc,
  "target/tmp/myGradientBoostingClassificationModel")
