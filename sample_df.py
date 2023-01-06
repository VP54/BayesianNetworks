# Databricks notebook source
def sample_df(df):

  df = df.sort_values(by = ['Target'], ascending = True)
  df_2 = df.sort_values(by = ['Target'], ascending = False)

  df_2 = df_2.head(10)
  df = df.head(100)

  return pd.concat([df, df_2])
