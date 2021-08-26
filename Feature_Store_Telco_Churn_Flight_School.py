# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://drive.google.com/uc?id=1-T_-28nMLT3YW9uiDl1hljv7dEZqmcv5" width=100/>
# MAGIC ### Before you begin...
# MAGIC 
# MAGIC __Run the cell below__ in order to create a text entry "widget" at the top of the screen.
# MAGIC   - To run a cell, click in the cell, then click the triangle on the upper right of the cell and choose "Run Cell" - alternatively, you can use SHIFT+ENTER.
# MAGIC 
# MAGIC Then, please __enter your email address__ or some other unique text string into the *user_email* text field above.  We use this to ensure that each Workshop user has unique file paths and a unique database name.  This allows multiple users to participate in the Workshop without stepping on each other's data, even if multiple users are logged in to Databricks with the same user ID.

# COMMAND ----------

# This creates the "user_email" field displayed at the top of the notebook.

dbutils.widgets.text("user_email", "Enter your email address");

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Store: Telco Churn example
# MAGIC 
# MAGIC This simple example will build a feature store on top of data from a telco customer churn data set.  The data set is based on original data from https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv, but pre-cleansed for this demo. 
# MAGIC 
# MAGIC We use the data to train a model and deploy both the model and features to production.
# MAGIC 
# MAGIC The goal is to produce a service that can predict in real-time whether a customer churns.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://drive.google.com/uc?id=1-T_-28nMLT3YW9uiDl1hljv7dEZqmcv5" width=100/>
# MAGIC ### Before you proceed...
# MAGIC 
# MAGIC __Run the cell below__ in order to set up your data sources and create a Delta Lake database for this module.
# MAGIC   - To run a cell, click in the cell, then click the triangle on the upper right of the cell and choose "Run Cell" - alternatively, you can use SHIFT+ENTER.

# COMMAND ----------

# DBTITLE 1,Housekeeping: Call Setup Notebook
# Call the setup notebook, which downloads data, and sets up local and dbfs file paths, and loads data to a Delta Lake database.  All of these will be unique for each workshop user.

user_id_raw = dbutils.widgets.get("user_email")

setup_responses = dbutils.notebook.run("./includes/setup", 0, {"user_email": user_id_raw}).split()

local_data_path = setup_responses[0]
dbfs_data_path = setup_responses[1]
dbfs_direct_data_path = setup_responses[2]
database_name = setup_responses[3]
table_name = setup_responses[4]

print(f"Path to be used for Local Files: {local_data_path}")
print(f"Path to be used for DBFS Files: {dbfs_data_path}")
print(f"Direct Path to be used for DBFS Files: {dbfs_direct_data_path}")
print(f"Database Name: {database_name}")
print(f"Raw input data Table Name: {table_name}")

# Set the default database name so we don't have to specify it every time
spark.sql(f"USE {database_name}")

# COMMAND ----------

# Now let's take a look at the input data we'll be using.

df_input = spark.sql(f"SELECT * FROM {database_name}.{table_name}")
display(df_input)

# COMMAND ----------

# Load our input table into a Dataframe
telco_df = spark.sql(f"SELECT * FROM {database_name}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining Feature Store Tables
# MAGIC 
# MAGIC This raw data set can be usefully improved for data science and machine learning by:
# MAGIC 
# MAGIC - Splitting it into logical, reusable subsets of columns
# MAGIC - Engineering more useful features
# MAGIC - Publishing the result as feature store tables
# MAGIC 
# MAGIC First we define a `@feature_table` that simply selects some demographic information from the data. This will become one feature store table when written later. A `@feature_table` is really just a function that computes a DataFrame defining the features in the table, from a source 'raw' DataFrame. It can be called directly for testing; this by itself does not persist or publish features.

# COMMAND ----------

from databricks.feature_store import feature_table

# These are the columns we want to use as a Demographic Features table
demographic_cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents"]

# 
# NOTE: A function that is decorated with the @feature_table decorator will gain these function attributes:
# 
# 	compute_and_write(
# 		input: Dict[str, Any],
# 		feature_table_name: str,
# 		mode: str = 'merge'
# 	) → pyspark.sql.dataframe.DataFrame
# 
# Note - Experimental: This function may change or be removed in a future release without warning.
# 

@feature_table
def compute_demographic_features(data):
  return data.select(demographic_cols)

demographics_df = compute_demographic_features(telco_df)
display(demographics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, features related to the customer's telco service are likewise selected. Note that each of these tables includes `customerID` as a key for joining. A little bit of feature engineering happens here - filling in 0 for `TotalCharges`.

# COMMAND ----------

@feature_table
def compute_service_features(data):
  #
  # These are the columns we want to use for a Service Features table
  #
  service_cols = ["customerID"] + [c for c in data.columns if c not in ["Churn"] + demographic_cols]
  return data.select(service_cols).fillna({"TotalCharges": 0.0})

service_df = compute_service_features(telco_df)
display(service_df)

# COMMAND ----------

# MAGIC %md
# MAGIC What's left? only the label, `Churn`. We could consider this a feature and write it in a feature store table. However in this use case, it's of course the very thing to be predicted, so that doesn't quite feel right.
# MAGIC 
# MAGIC Further, there may be other information _can_ be supplied at inference time, but does not make sense to consider a feature to _look up_. 
# MAGIC 
# MAGIC For example, in this case, it may be useful to know whether the customer's last call to customer service was escalated to a manager. Eventually, this information might become available and could be written into a feature store table for lookup. However, this model is probably going to be used to decide whether a customer on the phone who is asking for escalation is likely to churn _right now_. It's not possible to lookup whether the call _was escalated_ because it hasn't happened or not yet.
# MAGIC 
# MAGIC In this (fictional) example, the feature `LastCallEscalated` is "looked up" (really, it's just made up, but pretend!) because it could be important to making a predictive model. It and `Churn` however are not written as a feature store table. They are simply stored as a (normal) table. It's perfectly usable in training. At inference time, this kind of information (without of course the label `Churn`) will be supplied in a request to the model.

# COMMAND ----------

# Made up rule for LastCallEscalated: because ~26% of customers churn, if churned, then 35% chance call was escalated.
# Otherwise 15% chance.

import pyspark.sql.functions as F

inference_data_df = telco_df.select("customerID", "Churn").withColumn("LastCallEscalated",
    F.when(F.col("Churn"), F.hash(F.col("customerID")) % 100 < 35).otherwise(F.hash(F.col("customerID")) % 100 < 15))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing Feature Store Tables
# MAGIC 
# MAGIC With the tables defined by functions above, next, the feature store tables need to be written out, first as Delta tables. These are the fundamental 'offline' data stores underpinning the feature store tables. First (optionally) remove any existing feature stores - this would not happen in subsequent runs (but it's fine for this demo).

# COMMAND ----------

# UNCOMMENT THIS IF YOU NEED IT

# Temporary Developer API (proper API coming soon): manually clear out the feature table with a developer API
# NOTE: this API call works for DBR 8.x, but not for DBR 9.x, which requires another parameter which is shrouded in mystery
#
# NOTE: THIS WILL FAIL if the tables do not exist... but that's OK

#from databricks.feature_store import FeatureStoreClient

#fs = FeatureStoreClient()
#fs._catalog_client.delete_feature_table(f"{database_name}.demographic_features")
#fs._catalog_client.delete_feature_table(f"{database_name}.service_features")

# COMMAND ----------

# Drop these tables just in case they exist

spark.sql(f"DROP TABLE IF EXISTS {database_name}.demographic_features")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.service_features")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.inference_data")

# COMMAND ----------

# MAGIC %md
# MAGIC Next use the client to create the feature tables, defining metadata like which database and table the feature store table will write to, and importantly, its key(s).
# MAGIC 
# MAGIC The tables are created as Delta tables, and also registered in the Feature Store.

# COMMAND ----------

# MAGIC %md
# MAGIC It appears that the Feature Store exists per workspace... is this correct?  If so, is this desirable?

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Create the Demographic Features table
demographic_features_table = fs.create_feature_table(
  name=f'{database_name}.demographic_features',
  keys='customerID',
  schema=demographics_df.schema,
  description='Telco customer demographics')

# Create the Service Features table
service_features_table = fs.create_feature_table(
  name=f'{database_name}.service_features',
  keys='customerID',
  schema=service_df.schema,
  description='Telco customer services')

# COMMAND ----------

# MAGIC %md
# MAGIC The `@feature_table` functions themselves are 'called' to add or update data in the feature store table. The function gets a method `compute_and_write` from its decorator which takes in the source 'raw' DataFrame and computes and writes features.

# COMMAND ----------

# This is how we put new or updated information from the raw table into the Feature tables.
# Of course, this demo hasn't made any changes yet, but we want to show the API now.  We'll make changes later.

compute_demographic_features.compute_and_write(telco_df, feature_table_name=f"{database_name}.demographic_features")
compute_service_features.compute_and_write(telco_df, feature_table_name=f"{database_name}.service_features")

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, save the "inference data" including label and escalation status:

# COMMAND ----------

#
# Now we'll persist the inference data as a Delta table
#

spark.sql(f"DROP TABLE IF EXISTS {database_name}.inference_data")

inference_data_df.createOrReplaceTempView("vw_inference_data")
spark.sql(f"CREATE TABLE {database_name}.inference_data AS (SELECT * from vw_inference_data)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Browsing the Feature Store
# MAGIC 
# MAGIC The tables are now visible and searchable in the [Feature Store](/#feature-store) -- try it!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building Models from a Feature Store
# MAGIC 
# MAGIC Next, it's time to build a simple model predicting whether the customer churns. There are two feature store tables containing information we want to join to the other 'inference data' in order to build a model:
# MAGIC 
# MAGIC - `demographic_features`
# MAGIC - `service_features`
# MAGIC 
# MAGIC We could of course simply read the data from the tables as a DataFrame and join them. However, in a feature store, we want to define the joins declaratively, because the same join will need to happen in a quite different context in online serving. More on that later. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### An initial model
# MAGIC 
# MAGIC Perhaps modeling begins by building a model on the most immediately relevant features available - actual service-related features. Below, a `FeatureLookup` is created for each feature in the `service_features` table. This defines how other features are looked up (table, feature names) and based on what key -- `customerID`.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

# Instantiate a Feature Store client
fs = FeatureStoreClient()

# Get the Service Features Table
service_features_table = fs.get_feature_table(f"{database_name}.service_features")

# We'll call this function when we build the model
def generate_all_lookups(table):
  # Note that we are returning an anonymous FeatureLookup object
  # It contains all columns from the passed-in table except for customerID
  return [FeatureLookup(table.name, f, "customerID") for f in fs.read_table(table.name).columns if f != "customerID"]

# COMMAND ----------

# MAGIC %md
# MAGIC Now, modeling can proceed. `create_training_set` pulls together all the features. Its "input" is the `inference_data` table, as discussed above, and the `FeatureLookup`s define how to join in features. Modeling ignores the `customerID` of course; it's a key rather than useful data.
# MAGIC 
# MAGIC The modeling here is simplistic, and just trains a plain `sklearn` gradient boosting classifier.
# MAGIC 
# MAGIC Normally, this model would be logged (or auto-logged) by MLflow. It's necessary however to log to MLflow through the client's `log_model` method instead. The Feature Store needs to package additional metadata about feature lookups and feature tables in order for the model to perform lookups and joins correctly at inference time, when deployed as a service.
# MAGIC 
# MAGIC Note that code from `read_table` through to `log_model` must be inside an MLflow run. When complete, a new run has been logged and registered in the MLflow Model Registry.

# COMMAND ----------

import mlflow
import mlflow.shap
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.autolog(log_input_examples=True) # Optional

# Define a method for reuse later
def fit_model(model_feature_lookups):

  with mlflow.start_run():
    inference_data_df = spark.read.table(f"{database_name}.inference_data")
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="Churn", exclude_columns="customerID")

    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("Churn", axis=1)
    y = training_pd["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Churn is relatively rare; let's weight equally to start as it's probably at least as important as not-churning
    churn_weight = 1.0 / y_train.sum()
    not_churn_weight = 1.0 / (len(y) - y_train.sum())
    sample_weight = y_train.map(lambda churn: churn_weight if churn else not_churn_weight)

    # Not attempting to tune the model at all for purposes here
    gb_classifier = GradientBoostingClassifier(n_iter_no_change=10)
    # Need to encode categorical cols
    encoders = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == 'object'])])
    pipeline = Pipeline([("encoder", encoders), ("gb_classifier", gb_classifier)])
    pipeline_model = pipeline.fit(X_train, y_train, gb_classifier__sample_weight=sample_weight)

    mlflow.log_metric('test_accuracy', pipeline_model.score(X_test, y_test))
    # We save a lot of time by skipping this... mlflow.shap.log_explanation(gb_classifier.predict, encoders.transform(X))

    fs.log_model(
      pipeline_model,
      "model",
      flavor=mlflow.sklearn,
      training_set=training_set,
      registered_model_name="feature_store_telco_churn",
      input_example=X[:100],
      signature=infer_signature(X, y)) # signature logging may not work now
      
fit_model(generate_all_lookups(service_features_table))

# COMMAND ----------

# MAGIC %md
# MAGIC It's trivial to apply a registered MLflow model to features with `score_batch`. Again its only input are `customerID`s and `LastCallEscalated` (without the label `Churn` of course!). Everything else is looked up. However, eventually the goal is to produce a real-time service from this model and these features.

# COMMAND ----------

from pyspark.sql.functions import col

batch_input_df = inference_data_df.select("customerID", "LastCallEscalated")
with_predictions = fs.score_batch("models:/feature_store_telco_churn/1", batch_input_df, result_type='string')
with_predictions = with_predictions.withColumn("prediction", col("prediction") == "True")
display(with_predictions.join(inference_data_df.select("customerID", "Churn"), on="customerID"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### An improved model?
# MAGIC 
# MAGIC Of course, there were more features available above, demographic features. In the next iteration, maybe try adding those features?

# COMMAND ----------

demographic_features_table = fs.get_feature_table(f"{database_name}.demographic_features")
service_features_table = fs.get_feature_table(f"{database_name}.service_features")

fit_model(generate_all_lookups(service_features_table) + generate_all_lookups(demographic_features_table))

# COMMAND ----------

# MAGIC %md
# MAGIC The model doesn't improve much in accuracy, but perhaps slightly. A win? We can generate batch predictions again:

# COMMAND ----------

with_predictions = fs.score_batch("models:/feature_store_telco_churn/2", batch_input_df, result_type='string')
with_predictions = with_predictions.withColumn("prediction", col("prediction") == "True")
display(with_predictions.join(inference_data_df.select("customerID", "Churn"), on="customerID"))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice now in the feature store that the demographic features table shows up as used only by the second version of this model, not the first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Updating Feature Store Tables
# MAGIC 
# MAGIC Imagine sometime later, it's necessary to engineer more features. It's discovered that two more derived features would be useful to the model:
# MAGIC 
# MAGIC - `NumOptionalServices`: the count of optional services the customer has, like `StreamingTV` or `StreamingMovies`
# MAGIC - `AvgPriceIncrease`: the customer's current monthly charges, relative to their historical average monthly charge
# MAGIC 
# MAGIC `compute_service_features` is redefined to compute these new features:

# COMMAND ----------

@feature_table
def compute_service_features(data):
  # Count number of optional services enabled, like streaming TV
  @F.pandas_udf('int')
  def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('int'), cols))
  
  # Below also add AvgPriceIncrease: current monthly charges compared to historical average
  service_cols = ["customerID"] + [c for c in data.columns if c not in ["Churn"] + demographic_cols]
  return data.select(service_cols).fillna({"TotalCharges": 0.0}).\
    withColumn("NumOptionalServices",
        num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
    withColumn("AvgPriceIncrease",
        F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))

display(compute_service_features(telco_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, with the new definitions of features in hand, `compute_and_write` can add the new features by `merge`-ing them into the feature store table.

# COMMAND ----------

compute_service_features.compute_and_write(telco_df, feature_table_name=f"{database_name}.service_features", mode="merge")

# COMMAND ----------

# MAGIC %md
# MAGIC Try modeling again:

# COMMAND ----------

demographic_features_table = fs.get_feature_table(f"{database_name}.demographic_features")
service_features_table = fs.get_feature_table(f"{database_name}.service_features")

fit_model(generate_all_lookups(service_features_table) + generate_all_lookups(demographic_features_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publishing Feature Store Tables
# MAGIC 
# MAGIC So far, the features exist 'offline' as Delta tables. It's been very convenient to define features and compute them with Spark and persist them for batch use. However, these features need to be available 'online' too. It would be prohibitively slow recompute features in real-time. The features are already computed, but exist in tables which may be slow to read individual rows from; the Delta tables may not be accessible, even, from the real-time serving context!
# MAGIC 
# MAGIC The Feature Store can 'publish' the data in these tables to an external store that is more suitable for fast online lookups. `publish_table` does this.
# MAGIC 
# MAGIC **Note**: This will require the `oetrta-IAM-access` IAM role for RDS access.

# COMMAND ----------

# If you hit Run All, you will stop here.  
# This allows you to examine all artifacts before anything is deleted in the cell below.
# But don't forget to run the cell below, or you'll have a lot of manual cleanup to do!

dbutils.notebook.exit(0)

# COMMAND ----------

NONONO ORIGINAL DO NOT RUN

from databricks.feature_store.online_store_spec import AmazonRdsMySqlSpec
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

hostname = "oetrta-mysql.cmeifwaki1jl.us-west-2.rds.amazonaws.com" # dbutils.secrets.get("oetrta", "mysql-hostname")
port = 3306 # int(dbutils.secrets.get("oetrta", "mysql-port"))
user = "oetrtaadmin" # dbutils.secrets.get("oetrta", "mysql-username")
password = dbutils.secrets.get("oetrta", "mysql-password")

online_store = AmazonRdsMySqlSpec(hostname, port, user, password)

fs.publish_table(name='oetrta.demographic_features', online_store=online_store)
fs.publish_table(name='oetrta.service_features', online_store=online_store)

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonRdsMySqlSpec
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

hostname = "198.71.225.50" 
port = 3306 
user = "BKMySQL" 
password = "ownerMySQL!#1"

online_store = AmazonRdsMySqlSpec(hostname, port, user, password)

fs.publish_table(name=f'BK-MYSQL-SHARE-02.demographic_features', online_store=online_store)
fs.publish_table(name=f'BK-MYSQL-SHARE-02.service_features', online_store=online_store)

# COMMAND ----------

# MAGIC %md
# MAGIC This becomes visible in the feature table's UI as an "Online Store" also containing the same data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO: Model Serving
# MAGIC Not yet supported.

# COMMAND ----------

# MAGIC %md ## Aside: Save data for AutoML
# MAGIC 
# MAGIC After the Feature Store, we will walk through AutoML.  For that, let's save a table with both the features and the label which AutoML can use.

# COMMAND ----------

inference_data_df.join(service_df, "customerID").write.format("delta").mode("overwrite").saveAsTable(f"{database_name}.telco_automl")

# COMMAND ----------

# DBTITLE 1,Prevent "Run All" from running the cells below and deleting things
# If you hit Run All, you will stop here.  
# This allows you to examine all artifacts before anything is deleted in the cell below.
# But don't forget to run the cell below, or you'll have a lot of manual cleanup to do!

dbutils.notebook.exit(0)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://drive.google.com/uc?id=1-T_-28nMLT3YW9uiDl1hljv7dEZqmcv5" width=100/>
# MAGIC ### Before you leave...
# MAGIC 
# MAGIC Be sure to run the appropriate cells below.  They will clean up all your resources.

# COMMAND ----------

# DBTITLE 1,Clean up the Feature Store if you are done with it
# Temporary Developer API (proper API coming soon): manually clear out the feature table with a developer API
# NOTE: this API call works for DBR 8.x, but not for DBR 9.x, which requires another parameter which is shrouded in mystery
#
# NOTE: THIS WILL FAIL if the tables do not exist... but that's OK

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
fs._catalog_client.delete_feature_table(f"{database_name}.demographic_features")
fs._catalog_client.delete_feature_table(f"{database_name}.service_features")

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# DBTITLE 1,Clean up the MLflow Registry if you are done with it
# Cleanup... 
# Don't forget to run this cell, or you will have a lot of manual cleanup to do!
# NOTE: if you have Registered any of these models, you will have to un-register them before running this cell

import mlflow
import time
from mlflow.entities import ViewType

def delete_all_runs(run_infos):
    for r in run_infos:
        client.delete_run(r.run_id)

# Now you can delete the model
print("Deleting registered model...")
client.delete_registered_model('feature_store_telco_churn')
time.sleep(10) # the call above does not block, so leave time to ensure it finishes
print("... registered model deleted.")

# Now delete all runs
print("Deleting all runs...")
experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
delete_all_runs(mlflow.list_run_infos(experiment.experiment_id, run_view_type=ViewType.ALL)) # ViewType can be ACTIVE_ONLY, DELETED_ONLY, ALL
print("... all runs deleted.")


# COMMAND ----------

# DBTITLE 1,Be sure to run this cleanup!
# Cleanup... 
# Don't forget to run this cell, or you will have a lot of manual cleanup to do!
# Remove all data resources: local data path, dbfs data path, and Delta Lake database

cleanup_response = dbutils.notebook.run("./includes/cleanup", 0, {"user_email": user_id_raw})
print(cleanup_response)
