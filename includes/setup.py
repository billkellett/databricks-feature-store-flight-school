# Databricks notebook source
# MAGIC %md
# MAGIC ### Note to Demo Builders
# MAGIC 
# MAGIC This housekeeping notebook provides a template you can use to download and install the data sources for your demo.
# MAGIC 
# MAGIC The `user_email` widget above lets you test this notebook stand-alone, but it will normally be called by your main notebook.  When not testing stand-alone, be sure that the widget above is empty and this notebook is detached from your cluster.
# MAGIC 
# MAGIC This notebook makes it easy for you to build demos that have important advantages:
# MAGIC - It makes your demo completely portable, because it downloads its own data.  You can run it on any workspace, and even give it to customers to run in their environment
# MAGIC - It makes your demo safe for multi-user workshops, because each user gets a separate database, local file path, and DBFS file path, even if they are running on the same cluster with the same login.
# MAGIC - It removes housekeeping from the main notebook, which avoids distracting users with code that is not relevant to your topic.
# MAGIC 
# MAGIC You'll need to customize this notebook for your specific needs, but the template code below should make it easy for you.  There are comments in each cell that will help you pinpoint any changes you'll want to make.
# MAGIC 
# MAGIC We assume that all your data will be coming from a public folder on Google Drive.  The Knowledge Management team has made a shared folder you can use... km/km_public/demos.  You can create your own folder within, and make your data easily accessible.  

# COMMAND ----------

# NOTE to Demo Builders
# You probably won't have to make any changes in this cell.
# We accept user_email as a parameter passed from the main notebook, but you can also populate the widget above if you want to test this notebook standalone.
# We use the contents of user_email to generate unique local file paths, DBFS file paths, and databases for each user.

# Get the user's email address, which was passed in as a parameter by the notebook that called me.
# We'll strip special characters from this field, then use it to define unique path names (local and dbfs) as well as a unique database name.
# This prevents name collisions in a multi-user workshop.

# NOTE that it is possible to retrieve the User ID property programmatically.  
# However, I'm doing it with widgets just in case multiple workshop participants are using the same login.

dbutils.widgets.text("user_email", "");

# COMMAND ----------

###
### NOTE To Demo Builders...
### This code creates file paths beginning with "demo-" and a database beginning with "DEMO_"
### If you want the prefixes to be different, change them below in this cell.
###

# import to enable removal on non-alphanumeric characters (re stands for regular expressions)
import re

# Get the email address entered by the user on the calling notebook
user_id = dbutils.widgets.get("user_email")
print(f"Data entered in user_email field: {user_id}")

# Strip out special characters and make it lower case
clean_user_id = re.sub('[^A-Za-z0-9]+', '', user_id).lower();
print(f"User ID with special characters removed: {clean_user_id}");

# Construct the unique path to be used to store files on the local file system
local_data_path = f"feature-store-demo-{clean_user_id}/"
print(f"Path to be used for Local Files: {local_data_path}")

# Construct the unique path to be used to store files on the DBFS file system
dbfs_data_path = f"/dbfs/FileStore/feature-store-demo-{clean_user_id}/"
print(f"Path to be used for DBFS Files: {dbfs_data_path}")

# Make a DBFS path that does not use the FUSE mount
dbfs_direct_data_path = f"dbfs:/FileStore/feature-store-demo-{clean_user_id}/"
print(f"Direct Path to be used for DBFS Files: {dbfs_direct_data_path}")

# Construct the unique database name
database_name = f"FEATURE_STORE_DEMO_{clean_user_id}"
print(f"Database Name: {database_name}")

# COMMAND ----------

# NOTE to Demo Builders... you probably won't have to make any changes in this cell.

# Enables running shell commands from Python
# I need to do this, rather than just the %sh magic command, because I want to pass in python variables

import subprocess

# COMMAND ----------

# NOTE to Demo Builders... you probably won't have to make any changes in this cell.

# Delete local directories that may be present from a previous run

process = subprocess.Popen(['rm', '-f', '-r', local_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# NOTE to Demo Builders... you probably won't have to make any changes in this cell.

# Create local directories used in the workshop

process = subprocess.Popen(['mkdir', '-p', local_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# NOTE to Demo Builders... you probably won't have to make any changes in this cell.

# Delete DBFS directories that may be present from a previous run

process = subprocess.Popen(['rm', '-f', '-r', dbfs_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# NOTE to Demo Builders... you probably won't have to make any changes in this cell.

# Create DBFS directories used in the workshop

process = subprocess.Popen(['mkdir', '-p', dbfs_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use the cell below to download Google Drive files that are less than 100 MB.

# COMMAND ----------

# NOTE to Demo Builders... you'll want to change this cell to match your specific data download needs.
# This cell makes it easy to download files on Google Drive that are less than 100MB (for larger files, see the cell below).
# If you have multiple files, just clone this cell for each file to be downloaded and change the appropriate field values.

# Here is an example... just change the contents of output_name_of_small_file and google_drive_id, and it will work for you!

# This file is <100MB and Google Drive permits it to be downloaded in a simple way
# Just fill out the following fields before you run...

# This is the name you want to give the file after it is downloaded
# It can be different from the file name on Google Drive
output_name_of_small_file = 'Telco-Customer-Churn.csv'

# This is the ID of the file on Google Drive.  
# To get it, find the file on Google Drive, right-click it and select "Get Link."
# Make sure the file is marked as "Anyone on the internet with this link can view."
# The link will look something like this:
#   https://drive.google.com/file/d/1vlXV2YXLrKC-IUC4jywxJOrrDIlO_tEl/view?usp=sharing
# The ID is the part between /d/ and /view?  Pluck it out and use it below.
google_drive_id = '1p_Wy5lEwdBS22_vHKRzTU6_brWAAk8Dd'

#
# Now you can run this cell.  You don't have to change anything below.
# Your file will be downloaded into your dbfs_data_path
# However, if you want to download to your local path, just change "/dbfs/{dbfs_data_path}" to "{local_data_path}" below.
#

# If there is an old version of the data download out there, delete it
process = subprocess.Popen(['rm', f'{dbfs_data_path}{output_name_of_small_file}', '-f'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# Download the file
process = subprocess.Popen(['wget', '--no-check-certificate', f'https://docs.google.com/uc?export=download&id={google_drive_id}', f"--output-document={dbfs_data_path}{output_name_of_small_file}"],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# MAGIC %md
# MAGIC #### The cells below build database tables from the downloaded data.
# MAGIC 
# MAGIC You may want to delete or modify these cells, depending on your specific needs.

# COMMAND ----------

# Drop the user's database if present

spark.sql(f'DROP DATABASE IF EXISTS {database_name} CASCADE')

# COMMAND ----------

# Create an empty database for the user

spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name}')

# COMMAND ----------

#
# Now we'll massage the data a bit to make it more useful for this demo.
#

import pyspark.sql.functions as F

telco_df = spark.read.option("header", True).option("inferSchema", True).csv(f"{dbfs_direct_data_path}Telco-Customer-Churn.csv")

# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
# Yes/No -> boolean
for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
  telco_df = telco_df.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
# Contract categorical -> duration in months
telco_df = telco_df.withColumn("Contract",\
    F.when(F.col("Contract") == "Month-to-month", 1).\
    when(F.col("Contract") == "One year", 12).\
    when(F.col("Contract") == "Two year", 24))
# Empty TotalCharges -> NaN
telco_df = telco_df.withColumn("TotalCharges",\
    F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).\
    otherwise(F.col("TotalCharges").cast('double')))

telco_df.createOrReplaceTempView("vw_small_data")

# COMMAND ----------

# Set the new database as the default, so we don't have to keep specifying it.

spark.sql(f'USE {database_name}')

# COMMAND ----------

# Drop the database table just in case it exists

spark.sql(f"DROP TABLE IF EXISTS telco_customer_churn")

# COMMAND ----------

# Now create the table

spark.sql(f"""
    CREATE TABLE telco_customer_churn
    USING DELTA
    AS (
        SELECT *
        FROM vw_small_data
       )""")

# COMMAND ----------

# NOTE to Demo Builders... You probably don't need to change this cell,
# unless you want to return other data to the main notebook. 
# Note that we return a string (that's the only choice) with space-separated values.
# The main notebook parses the string and extracts the values.

# Return to the caller, passing the variables needed for file paths and database

response = local_data_path + " " + dbfs_data_path + " " + dbfs_direct_data_path + " " + database_name + " " + 'telco_customer_churn'

dbutils.notebook.exit(response)

# COMMAND ----------


