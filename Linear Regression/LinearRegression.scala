import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._
import spark.implicits._

// NOTE: Code not packed into an object for easier access and manipulation with Spark shell for learning purposes

// Setting error reporting to remove excess error notices
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

//reading in .csv regarding users' ecommerce data
//  This linear regression will attempt to create a model to predict the amount spent per year for customers
//  based on various factors like time on app, length of Membership, etc.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean_Ecommerce.csv")
//viewing a bit of sample data
data.printSchema
data.head(3)

//Grabbing the useful quantitative data from the csv and setting the YAS as the label
val df = data.select(data("Yearly Amount Spent").as("label"),$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")
//reassembling the data into the (label,features) format that Scala ML library uses for modeling
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
val out = assembler.transform(df).select($"label",$"features")

val linreg = new LinearRegression()
val lrModel = linreg.fit(out)

// Printing model data
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
val trainingSummary = lrModel.summary
// residuals, RMSE, MSE, and R^2 Values.
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

//Potential improvements:
//Using ElasticNet regularization for better/more accurate general-case fitting
//Creating a test-train split to avoid overfitting and to create a testing dataset to confirm accuracy of model
