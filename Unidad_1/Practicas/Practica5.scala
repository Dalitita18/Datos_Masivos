import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")
df.printSchema()
//1
df.groupBy("Company").mean().show()
//2
df.groupBy("Company").count().show()
//3
df.groupBy("Company").max().show()
//4
df.groupBy("Company").min().show()
//5
df.groupBy("Company").sum().show()
//6
df.select(countDistinct("Sales")).show()
//7
df.select(sumDistinct("Sales")).show()
//8
df.select(variance("Sales")).show()
//9
df.select(stddev("Sales")).show()
//10
df.select(collect_set("Sales")).show()
//11
df.groupBy("Person").mean().show()
//12
df.groupBy("Person").count().show()
//13
df.groupBy("Person").max().show()
//14
df.groupBy("Person").min().show()
//15
df.groupBy("Person").sum().show()
//16
df.groupBy($"Company" === "GOOG").mean().show()
//17
df.groupBy($"Company" === "FB").mean().show()
//18
df.groupBy($"Company" === "MSFT").mean().show()
//19
df.groupBy($"Company" === "GOOG").min().show()
//20
df.groupBy($"Company" === "GOOG").max().show()