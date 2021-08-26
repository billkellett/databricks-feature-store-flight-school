# Databricks notebook source
# MAGIC %md
# MAGIC ### Note to Demo Builders...
# MAGIC 
# MAGIC This cleanup notebook deletes the user's:
# MAGIC - Local file path
# MAGIC - DBFS file path
# MAGIC - Database
# MAGIC 
# MAGIC The `user_email` widget above lets you test this notebook stand-alone, but it will normally be called by your main notebook.  When not testing stand-alone, be sure that the widget above is empty and this notebook is detached from your cluster.
# MAGIC 
# MAGIC You probably won't have to change any of the existing cells, but you may want to add code for any special cleanup requirements you have.  For example, if you are using MLflow you might want to delete your experiment. 

# COMMAND ----------

# Get the user's email address, which was passed in as a parameter by the notebook that called me.
# We'll strip special characters from this field, then use it to define unique path names (local and dbfs) as well as a unique database name.
# This prevents name collisions in a multi-user workshop.

# NOTE that it is possible to retrieve the User ID property programmatically.  
# However, I'm doing it with widgets just in case multiple workshop participants are using the same login.

dbutils.widgets.text("user_email", "");

# COMMAND ----------

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
database_name = f"feature_store_demo_{clean_user_id}"
print(f"Database Name: {database_name}")

# COMMAND ----------

# Enables running shell commands from Python
# I need to do this, rather than just the %sh magic command, because I want to pass in python variables

import subprocess

# COMMAND ----------

# Drop the user's database 

spark.sql(f'DROP DATABASE IF EXISTS {database_name} CASCADE')

# COMMAND ----------

# Delete local directories created for workshop

process = subprocess.Popen(['rm', '-f', '-r', local_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Delete DBFS directories created for workshop

process = subprocess.Popen(['rm', '-f', '-r', dbfs_data_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Return to the caller

response = "Cleanup is complete"

dbutils.notebook.exit(response)
