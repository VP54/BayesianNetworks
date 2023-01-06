# Databricks notebook source
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

def label_categorical_variables(df, column_lst):
  
  for column in column_lst:
    enc = LabelEncoder()
    transformed = enc.fit(df[column]) 
    df[column] = enc.transform(df[column])
    
  return df

# COMMAND ----------

def label_trim():
  
  df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
  column_lst = ['Type', 'Product ID']
  df = label_categorical_variables(df, column_lst)
  
  return df.drop(columns=['UDI', 'Product ID', 'Failure Type'])

# COMMAND ----------

df = label_trim()
df.to_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

# COMMAND ----------

display(df)