# Databricks notebook source
from random import seed
from random import randint

# COMMAND ----------

def crete_missing_values(df):
  
  col_lst = ['Teplota vzduchu', 'Provozní teplota', 'Rychlost otáček']
  missing_values = []

  for name in col_lst:
    for _ in range(500):
        missing_values.append(randint(0, len(df) - 1))
        for value in missing_values:
          df[name].loc[value] == np.nan
  
  return df