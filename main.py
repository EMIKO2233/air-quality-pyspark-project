from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

# Create Spark session
spark = SparkSession.builder \
    .appName("AirQualityProject") \
    .getOrCreate()

# Load dataset
df = spark.read.csv("data/Bexley_Belvedere.csv", header=True)

# Convert the datetime column after previous load showed strings
df = df.withColumn(
    "datetime",
    to_timestamp("datetime", "dd/MM/yyyy HH:mm")
)

# Convert the numeric columns after previous load showed strings, using double rather than float (greater accuracy)
df = df.withColumn("pm25", col("pm25").cast("double")) \
       .withColumn("pm10", col("pm10").cast("double")) \
       .withColumn("wind_speed", col("wind_speed").cast("double")) \
       .withColumn("wind_dir", col("wind_dir").cast("double"))

# Show data
df.show(5)

# Show the schema
df.printSchema()
