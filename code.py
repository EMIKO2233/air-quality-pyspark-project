from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql.functions import count, when
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import hour, dayofweek

# Creating a spark session
spark = SparkSession.builder \
    .appName("AirQualityProject") \
    .getOrCreate()

# Loading the dataset
df = spark.read.csv("data/Bexley15min.csv", header=True)

# Renaming columns
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

# Convert numeric data columns
df = df.withColumn("pm25", col("pm25").cast("double")) \
       .withColumn("pm10", col("pm10").cast("double")) \
       .withColumn("wind_speed", col("wind_speed").cast("double")) \
       .withColumn("wind_dir", col("wind_dir").cast("double"))

df.show(5)

# Show the schema
df.printSchema()

#---------------------------------------------------------------------
# Exploring the data and cache
#---------------------------------------------------------------------

# Missing values
missing_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])
missing_counts.show()

# Handle missing values
df = df.dropna()

# Cache
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


#------------------------------------------------------------------------
# Visualisation - Combined 4 Plot Grid of Wind and particulates
#------------------------------------------------------------------------

sample_df = df.sample(fraction=0.01).toPandas()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1️ Wind Speed vs PM2.5
axes[0, 0].scatter(sample_df["wind_speed"], sample_df["pm25"], color='blue')
axes[0, 0].set_xlabel("Wind Speed")
axes[0, 0].set_ylabel("PM2.5")
axes[0, 0].set_title("Wind Speed vs PM2.5")

# 2️ Wind Speed vs PM10
axes[0, 1].scatter(sample_df["wind_speed"], sample_df["pm10"], color='green')
axes[0, 1].set_xlabel("Wind Speed")
axes[0, 1].set_ylabel("PM10")
axes[0, 1].set_title("Wind Speed vs PM10")

# 3 Wind Direction vs PM2.5
axes[1, 0].scatter(sample_df["wind_dir"], sample_df["pm25"], color='blue')
axes[1, 0].set_xlabel("Wind Direction")
axes[1, 0].set_ylabel("PM2.5")
axes[1, 0].set_title("Wind Direction vs PM2.5")

# 4️ Wind Direction vs PM10
axes[1, 1].scatter(sample_df["wind_dir"], sample_df["pm10"], color='green')
axes[1, 1].set_xlabel("Wind Direction")
axes[1, 1].set_ylabel("PM10")
axes[1, 1].set_title("Wind Direction vs PM10")

plt.tight_layout()

plt.show()


#---------------------------------------------------------------------
# Saving as parquet --this causing issue as follows
#HADOOP_HOME and hadoop.home.dir are unsetwinutils.exe not found
#---------------------------------------------------------------------

#df.write.mode("overwrite").parquet("data/air_quality_clean.parquet")

# Reload (best practice for next stages)
#df = spark.read.parquet("data/air_quality_clean.parquet")

#---------------------------------------------------------------------
#Feature Engeneering
#---------------------------------------------------------------------

df = df.withColumn("hour", hour("datetime")) \
       .withColumn("day_of_week", dayofweek("datetime"))

#Train Test split 80/20
train_df = df.filter(df.datetime < "2022-01-01")
test_df = df.filter(df.datetime >= "2022-01-01")

# define features and label columns
feature_cols = ["wind_speed", "wind_dir", "hour", "day_of_week"]
label_col = "pm25"
#ADDED: Second label for PM10
label_col_pm10 = "pm10"

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


#ADDED:Random Forest for PM10
rf_pm10 = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col_pm10
)

pipeline_rf_pm10 = Pipeline(stages=[
    assembler,
    rf_pm10
])

#------------------------------------------------------------------------
# Training the Models and Make Predictions
#------------------------------------------------------------------------

# Train LR and DT normally
model_lr = pipeline_lr.fit(train_df)
model_dt = pipeline_dt.fit(train_df)

pred_lr = model_lr.transform(test_df)
pred_dt = model_dt.transform(test_df)

#------------------------------------------------------------------------
# Hyperparameter Tuning 2.5 PM (Cross Validation of Random Forest)
#------------------------------------------------------------------------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Evaluator 
evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# CrossValidator
crossval = CrossValidator(
    estimator=pipeline_rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# Train tuned model
cv_model = crossval.fit(train_df)

# Predictions using tuned model
pred_rf = cv_model.transform(test_df)

# Evaluate tuned random forrest
rmse_rf = evaluator.evaluate(pred_rf)
print("\nPM2.5 Tuned Random Forest RMSE:", rmse_rf)

# Get the best model
best_model = cv_model.bestModel
best_rf = best_model.stages[-1]

print("\nBest Parameters:")
print("numTrees:", best_rf.getNumTrees)
print("maxDepth:", best_rf.getOrDefault("maxDepth"))


#------------------------------------------------------------------------
# Hyperparameter Tuning (PM10)
#------------------------------------------------------------------------

# Parameter grid for PM10
paramGrid_pm10 = ParamGridBuilder() \
    .addGrid(rf_pm10.numTrees, [20, 50]) \
    .addGrid(rf_pm10.maxDepth, [5, 10]) \
    .build()

# CrossValidator for PM10
crossval_pm10 = CrossValidator(
    estimator=pipeline_rf_pm10,
    estimatorParamMaps=paramGrid_pm10,
    evaluator=RegressionEvaluator(
        labelCol=label_col_pm10,
        predictionCol="prediction",
        metricName="rmse"
    ),
    numFolds=3
)

# Train PM10 model
cv_model_pm10 = crossval_pm10.fit(train_df)

# Predictions
pred_rf_pm10 = cv_model_pm10.transform(test_df)

# Evaluate PM10
evaluator_pm10 = RegressionEvaluator(
    labelCol=label_col_pm10,
    predictionCol="prediction",
    metricName="rmse"
)

rmse_rf_pm10 = evaluator_pm10.evaluate(pred_rf_pm10)

print("\nPM10 Tuned Random Forest RMSE:", rmse_rf_pm10)

# Best model PM10
best_model_pm10 = cv_model_pm10.bestModel
best_rf_pm10 = best_model_pm10.stages[-1]

print("\nPM10 Best Parameters:")
print("numTrees:", best_rf_pm10.getNumTrees)
print("maxDepth:", best_rf_pm10.getOrDefault("maxDepth"))

#------------------------------------------------------------------------
# Feature Importance (PM2.5 - Random Forest)
#------------------------------------------------------------------------

rf_model = best_model.stages[-1]

importances = rf_model.featureImportances
feature_importance = list(zip(feature_cols, importances))

print("\nPM2.5 Feature Importances:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")

#------------------------------------------------------------------------
# Feature Importance (PM10 - Random Forest)
#------------------------------------------------------------------------

rf_model_pm10 = best_model_pm10.stages[-1]

importances_pm10 = rf_model_pm10.featureImportances
feature_importance_pm10 = list(zip(feature_cols, importances_pm10))

print("\nPM10 Feature Importances:")
for feature, importance in feature_importance_pm10:
    print(f"{feature}: {importance:.4f}")

#------------------------------------------------------------------------
# Model Evaluation
#------------------------------------------------------------------------

rmse_lr = evaluator.evaluate(pred_lr)
rmse_dt = evaluator.evaluate(pred_dt)

print("\nPM2.5 Models:")
print("Linear Regression RMSE:", rmse_lr)
print("Decision Tree RMSE:", rmse_dt)
print("Random Forest RMSE:", rmse_rf)

print("\nPM10 Model:")
print("Random Forest RMSE:", rmse_rf_pm10)

#------------------------------------------------------------------------
# Show Predictions 
#------------------------------------------------------------------------

print("\nPM2.5 Predictions:")
pred_rf.select("pm25", "prediction").show(5)

print("\nPM10 Predictions:")
pred_rf_pm10.select("pm10", "prediction").show(5)
