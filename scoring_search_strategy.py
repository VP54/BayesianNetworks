# Databricks notebook source
# MAGIC %run ./Plotter

# COMMAND ----------

def Search_greedily(df, method):
  gs = HillClimbSearch(df)
  plot(gs.estimate(scoring_method=BicScore(df)), node_sizes),
  
  return gs.estimate(scoring_method=BicScore(df))