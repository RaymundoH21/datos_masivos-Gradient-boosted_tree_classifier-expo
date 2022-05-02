![LOGO](IMG/LOGO.png)
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
-------

### Ejemplo  
  
```scala
// scalastyle:off println
package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
// $example off$

object GradientBoostingClassificationExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GradientBoostedTreesClassificationExample")
    val sc = new SparkContext(conf)
    // $example on$
    //Cargar el codigo del archivo de datos
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    //Separar la informacion en instrucciones y en sets de prueba (30% se mantendra fuera de la prueba)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // instruccion del modelo GradientBoostedTrees.
    // El defaultParams de la clasificacion usa Logloss por default
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3 // Nota: usar mas iteraciones en la practica.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    // categoricalFeaturesInfo vacias indican que todas las funcione son continuas.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluar el modelo de prueba en instancias de prueba y test de error de computadora.
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification GBT model:\n ${model.toDebugString}")

    // Guardar y cargar el modelo.
    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    val sameModel = GradientBoostedTreesModel.load(sc,
      "target/tmp/myGradientBoostingClassificationModel")
    // $example off$

    sc.stop()
  }
}
// scalastyle:on println