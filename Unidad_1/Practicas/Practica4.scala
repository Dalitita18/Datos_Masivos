//Operaciones con el df
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

//1
df.show()
//2
df.filter($"Close" < 480 && $"High" < 480).show()
//3
df.select(corr("High", "Low")).show()
//4
df.columns
//5
df.count
//6
df.select($"Close" < 500 && $"High" < 600).count()
//7
df.select(sum("High")).show()
//8
df.select(min("High")).show()
//9
df.select(max("High")).show()
//10
df.select(mean("High")).show()
//11
df.select("High").first()
//12
df.first()
//13
df.describe()
//14
df.sort()
//15
df.("High")
//16
df.printSchema()
//17
df.select(year(df("Date"))).show()
//18
df.select(month(df("Date"))).show()
//19
df.filter($"High" > 480).count()
//20
df.filter($"High"===484.40).show()