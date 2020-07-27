
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Setting error reporting to reduce warnings
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

// Reading in advertising data to determine who clicked on an ad based on some data
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

// Printing some sample data to get a feel for it
data.printSchema()
data.head(1)

// Setting up dataframe
val hourdata = data.withColumn("Hour",hour(data("Timestamp")))
val cleandata = hourdata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
val cleandatafinal = cleandata.na.drop()

// Creating (label,features) format dataframe
val assembler = new VectorAssembler().setInputCols(Array("label","Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")

// Train-test split of data
val Array(training,test) = cleandatafinal.randomSplit(Array(0.7,0.3))

// Pipeline and LogReg model setup and usage
val lr = new LogisticRegression()
val pipe = new Pipeline().setStages(Array(assembler,lr))
val model = pipe.fit(training)
val result = model.transform(test)

// Model eval
// Must convert the test results to an RDD for MulticlassMetrics object
val predictionAndLabels = result.select($"prediction",$"label").as[(Double, Double)].rdd
val metric = new MulticlassMetrics(predictionAndLabels)
// Basic confusion matrix to analyze result of LogReg model
println("Confusion matrix below: ")
println(metric.confusionMatrix)
