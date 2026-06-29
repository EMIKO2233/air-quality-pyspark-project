from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

# Creating a spark session
spark = SparkSession.builder \
    .appName("AirQualityProject") \
    .getOrCreate()

# Loading the dataset
df = spark.read.csv("data/Bexley15min.csv", header=True)

# Rename columns
df = df.withColumnRenamed("ReadingDateTime", "datetime") \
       .withColumnRenamed("2.5", "pm25") \
       .withColumnRenamed("10", "pm10") \
       .withColumnRenamed("Wdir", "wind_dir") \
       .withColumnRenamed("Wspeed", "wind_speed")

# Convert datetime
df = df.withColumn(
    "datetime",
    to_timestamp("datetime", "dd/MM/yyyy HH:mm")
)

# Convert numeric columns
df = df.withColumn("pm25", col("pm25").cast("double")) \
       .withColumn("pm10", col("pm10").cast("double")) \
       .withColumn("wind_speed", col("wind_speed").cast("double")) \
       .withColumn("wind_dir", col("wind_dir").cast("double"))

# Show data
df.show(5)

# Show schema
df.printSchema()

#---------------------------------------------------------------------
# Exploring the data and cache
#---------------------------------------------------------------------

from pyspark.sql.functions import count, when

# Missing values
missing_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])
missing_counts.show()

# Handle missing values
df = df.dropna()

# Cache after cleaning
df.cache()
df.count()

# Confirm missing values are gone
df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()

# Summary statistics
df.describe().show()

# Correlations
print("pm25 vs wind_speed:", df.stat.corr("pm25", "wind_speed"))
print("pm25 vs wind_dir:", df.stat.corr("pm25", "wind_dir"))
print("pm10 vs wind_speed:", df.stat.corr("pm10", "wind_speed"))
print("pm10 vs wind_dir:", df.stat.corr("pm10", "wind_dir"))

#---------------------------------------------------------------------
# Saving as parquet ---this causing issue as HADOOP_HOME and hadoop.home.dir are unset
#winutils.exe not found
#---------------------------------------------------------------------

df.write.mode("overwrite").parquet("data/air_quality_clean.parquet")

# Reload (best practice for next stages)
df = spark.read.parquet("data/air_quality_clean.parquet")

#---------------------------------------------------------------------
#Feature engeneering
#---------------------------------------------------------------------

from pyspark.sql.functions import hour, dayofweek

df = df.withColumn("hour", hour("datetime")) \
       .withColumn("day_of_week", dayofweek("datetime"))

#Train Test split 80/20
train_df = df.filter(df.datetime < "2022-01-01")
test_df = df.filter(df.datetime >= "2022-01-01")

# define features and label columns
feature_cols = ["wind_speed", "wind_dir", "hour", "day_of_week"]
label_col = "pm25"

#--------------------------------------------------------------------
# Importing the machine learning tools
#--------------------------------------------------------------------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml import Pipeline

#---------------------------------------------------------------------
# VectorAssembler and Scaler
#---------------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
# scaler

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

#------------------------------------------------------------------------
#Build the ML models
#------------------------------------------------------------------------

#Linear Regression
lr = LinearRegression(
    featuresCol="scaled_features",
    labelCol=label_col
)

pipeline_lr = Pipeline(stages=[
    assembler,
    scaler,
    lr
])

#Decision Tree
dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol=label_col
)

pipeline_dt = Pipeline(stages=[
    assembler,
    dt
])

#Random Forest
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col
)

pipeline_rf = Pipeline(stages=[
    assembler,
    rf
])

#------------------------------------------------------------------------
# Training the Models and Make Predictions
#------------------------------------------------------------------------

model_lr = pipeline_lr.fit(train_df)
model_dt = pipeline_dt.fit(train_df)
model_rf = pipeline_rf.fit(train_df)

pred_lr = model_lr.transform(test_df)
pred_dt = model_dt.transform(test_df)
pred_rf = model_rf.transform(test_df)

#------------------------------------------------------------------------
# Model Evaluation
#------------------------------------------------------------------------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)

rmse_lr = evaluator.evaluate(pred_lr)
rmse_dt = evaluator.evaluate(pred_dt)
rmse_rf = evaluator.evaluate(pred_rf)

print("Linear Regression RMSE:", rmse_lr)
print("Decision Tree RMSE:", rmse_dt)
print("Random Forest RMSE:", rmse_rf)


pred_rf.select("pm25", "prediction").show(5)
