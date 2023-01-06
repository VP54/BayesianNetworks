# Databricks notebook source
import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Target"]

# COMMAND ----------

class SubSample:
  """
  Subsamples Dataframe with respect to relative counts
  """

  def __init__(self, df, SAMPLE_SIZE):
    self.df = df
    self.SAMPLE_SIZE = SAMPLE_SIZE
  
  def _match_bins(self, STEP_SIZE, col):
    
    df_perc =  [
      np.percentile(self.df[col], perc, axis = 0)
      for perc in np.arange(start = STEP_SIZE, stop=100, step = STEP_SIZE)
    ]
    
    return np.digitize(self.df[col], df_perc)
    
  def _weigh_bins(self, STEP_SIZE, col):
    
    bins = self._match_bins(STEP_SIZE, col)
    len_bins = np.bincount(bins)
    
    abs_weighs = 1 / np.array([len_bins[bin] for bin in bins]) 
    
    return abs_weighs / sum(abs_weighs)

    
  def _subsample_column(self, col, BINS = 100):
    
    STEP_SIZE = 100 / BINS

    bin_prob = self._weigh_bins(STEP_SIZE, col)
    
    return np.random.choice(self.df[col], size = self.SAMPLE_SIZE, p = bin_prob, replace = False)
    
    
  def _subsample_df(self):
    
    sampled_df = pd.DataFrame()
    
    for col in self.df.columns:
      if col == "Target":
        
          sampled_df[col] = self._subsample_column(col, 2)
      sampled_df[col] = self._subsample_column(col)
    
    return sampled_df

sample = SubSample(df = df, SAMPLE_SIZE = 500)
sampled_df = sample._subsample_df()


# COMMAND ----------

len(sampled_df[sampled_df['Target'] == 0]) / 500

# COMMAND ----------

len(df[df['Target'] == 1])

# COMMAND ----------

339 / 10000

# COMMAND ----------

from typing import Optional, Tuple
def histplot( 
              data, 
              bins, 
              x_name: str, 
              title: str, 
              text: Optional[str] = None, 
              figsize: Optional[Tuple[int, int]] = None,
              distribution: Optional[str] = None,
              tol: Optional[str] = None,
              name: Optional[str] = None
        ):

  if figsize is not None:
    fig, ax = plt.subplots(figsize = figsize)
  else:
    fig, ax = plt.subplots(figsize = (20, 10))

  n, bins, patches = plt.hist(data, 50, density=False, facecolor='b', alpha=0.75)

  if (self.SMALL_SIZE is not None) and (self.SMALL_SIZE is not None):
    self._config_text(SMALL_SIZE = self.SMALL_SIZE, BIG_SIZE = self.BIG_SIZE)

  #TEXT GRAPH
  plt.xlabel(x_name)
  plt.ylabel('Počet výskytů náhodné veličiny')

  plt.title(title)

  if text is not None:
    plt.text(text)

  plt.grid(True)

  self.save_figure(
                distribution = distribution,
                tol = tol,
                name = name
              )

  return self.save_figure(
                distribution = distribution,
                tol = tol,
                name = name
              )

# COMMAND ----------

import seaborn as sns 

histplot(df['Target'], bins = 100, x_name = '', title = '')

# COMMAND ----------

class SubSample:
  """
  Subsamples Dataframe with respect to relative counts
  """

  def __init__(self, df, SAMPLE_SIZE):
    self.df = df
    self.SAMPLE_SIZE = SAMPLE_SIZE
  
  def _match_bins(self, STEP_SIZE, col):
    
    df_perc =  [
      np.percentile(self.df[col], perc, axis = 0)
      for perc in np.arange(start = STEP_SIZE, stop=100, step = STEP_SIZE)
    ]

    return np.digitize(self.df[col], df_perc)
    
  def _weigh_bins(self, STEP_SIZE, col):
    
    bins = self._match_bins(STEP_SIZE, col)
    len_bins = np.bincount(bins)
    
    abs_weighs = 1 / np.array([len_bins[bin] for bin in bins]) 
    
    return abs_weighs / sum(abs_weighs)

    
  def _subsample_column(self, col, BINS = 100):
    
    STEP_SIZE = 100 / BINS

    bin_prob = self._weigh_bins(STEP_SIZE, col)
    
    return np.random.choice(self.df[col], size = self.SAMPLE_SIZE, p = bin_prob, replace = False)
    
    
  def _subsample_df(self):
    
    sampled_df = pd.DataFrame()
    
    for col in self.df.columns:
      if col == "Target":
          sampled_df[col] = self._subsample_column(col, 500)
      sampled_df[col] = self._subsample_column(col)
    
    return sampled_df

sample = SubSample(df = df, SAMPLE_SIZE = 800)
sampled_df = sample._subsample_df()


import seaborn as sns 

histplot(sampled_df['Target'], bins = 2, x_name = '', title = '')

# COMMAND ----------

len(sampled_df[sampled_df['Target'] == 1]) / len(sampled_df)

# COMMAND ----------

len(df[df['Target'] == 1]) / len(df)

# COMMAND ----------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.03, random_state=42)

# COMMAND ----------

histplot(y_test, bins = 2, x_name = '', title = '')

# COMMAND ----------



histplot(x_test['Rotational speed [rpm]'], bins = 100, x_name = '', title = '')

# COMMAND ----------



histplot(x_train['Rotational speed [rpm]'], bins = 100, x_name = '', title = '')

# COMMAND ----------



histplot(x_test['Torque [Nm]'], bins = 100, x_name = '', title = '')

# COMMAND ----------



histplot(x_train['Torque [Nm]'], bins = 100, x_name = '', title = '')