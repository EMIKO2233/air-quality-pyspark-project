# Air Quality Prediction Using PySpark

## Overview

This project investigates the relationship between wind properties and particulate air pollution concentrations (PM2.5 and PM10) using machine learning techniques within a distributed PySpark environment.

The analysis uses air-quality and meteorological data from the London Air Quality Network (LAQN), focusing on measurements collected at the Bexley monitoring station between 2018 and 2025.

## Objectives

- Analyse relationships between wind speed, wind direction and pollution concentrations.
- Develop regression models to predict PM2.5 and PM10.
- Demonstrate distributed data processing using PySpark.
- Apply machine learning pipelines and model evaluation techniques.
- Explore feature importance and environmental influences on air quality.

## Technologies Used

- Python
- PySpark DataFrame API
- PySpark MLlib
- Pandas
- Matplotlib

## Machine Learning Models

The following models were implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Additional machine learning components included:

- Feature Engineering
- VectorAssembler
- StandardScaler
- Pipeline Construction
- Cross Validation
- Hyperparameter Tuning
- Regression Evaluation using RMSE

## Data Processing Workflow

1. Load and preprocess LAQN data using Spark DataFrames.
2. Rename and convert variables to appropriate data types.
3. Identify and remove missing values.
4. Cache the cleaned dataset to improve performance.
5. Perform exploratory data analysis and visualisation.
6. Create temporal features (`hour` and `day_of_week`).
7. Split the dataset into training and testing sets using a time-based approach.
8. Train and evaluate machine learning models.
9. Analyse feature importance.

## Testing

Spark-based validation tests were implemented to verify:

- Schema validation
- Missing value validation
- Feature engineering validation
- Train/test split validation

Due to local environment restrictions preventing installation of the PyArrow dependency required by `pyspark.testing.utils`, assertion-based Spark tests were used in the main implementation. Equivalent examples using `assertSchemaEqual()` and `assertDataFrameEqual()` are included in the project documentation.

## Key Results

### PM2.5 Model Performance

| Model | RMSE |
|--------|------|
| Linear Regression | 6.80 |
| Decision Tree | 6.59 |
| Random Forest | 6.60 |

### PM10 Model Performance

| Model | RMSE |
|--------|------|
| Random Forest | 11.36 |

### Feature Importance

For both PM2.5 and PM10, wind direction was identified as the most important predictor, followed by wind speed.

## Dataset

Source: London Air Quality Network (LAQN)

Variables used:

- PM2.5 Concentration
- PM10 Concentration
- Wind Speed
- Wind Direction
- Timestamp

Location:

- Bexley Monitoring Station, London

Period:

- 2018–2025

## Running the Project

### Prerequisites

- Python 3.x
- Apache Spark
- PySpark
- Pandas
- Matplotlib

## Future Improvements

Potential future enhancements include:

- Incorporating additional environmental variables such as temperature, humidity and rainfall.
- Analysing multiple monitoring stations across London.
- Evaluating more advanced machine learning models.
- Improving prediction of extreme pollution events.
- Integrating real-time data through API ingestion.


