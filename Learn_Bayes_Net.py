# Databricks notebook source
pip install pgmpy

# COMMAND ----------

import pandas as pd
import numpy as np


from pgmpy.estimators import HillClimbSearch, BicScore, PC

# COMMAND ----------

# MAGIC %run ./ML_score

# COMMAND ----------

# MAGIC %run ./scoring_search_strategy

# COMMAND ----------

# MAGIC %run ./sample_df

# COMMAND ----------

# MAGIC %run ./create_MCAR

# COMMAND ----------

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df.drop(columns = ['Unnamed: 0', 'Type', 'Process temperature [K]'])
df = df.head(3000)

#df = sample_df(df)
df.columns = ['Teplota vzduchu', 'Rychlost otáček', 'Kroutící moment', 'Opotřebení nástroje', 'Porucha']
node_sizes = 5000

# COMMAND ----------

import pandas as pd

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df.drop(columns = ['Unnamed: 0', 'Type', 'Process temperature [K]'])
df.columns = ['Teplota vzduchu', 'Rychlost otáček', 'Kroutící moment', 'Opotřebení nástroje', 'Porucha']

df[df['Porucha'] == 1].reset_index(drop = True) 

from sklearn.model_selection import train_test_split

_, x_test, _, y_test = train_test_split(
                                                     df.loc[:, df.columns != 'b'], 
                                                     df['Porucha'], 
                                                     test_size=0.99, 
                                                     random_state=42
                                                   )

x_test['Porucha'] = y_test
x_test

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Učení struktury Bayesovské sítě dvěmi způsoby:
# MAGIC 
# MAGIC 1. Hladovým algoritmem
# MAGIC 2. Hill Climbing Algoritmem

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1. Hladový algoritmus

# COMMAND ----------

'''bic_model = Search_greedily(df, BicScore(df))

plot(bic_model, node_sizes)'''

# COMMAND ----------

import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

# COMMAND ----------

from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore, K2Score, BDsScore

x_test['Teplota vzduchu'] = x_test['Teplota vzduchu'].apply(lambda x: np.float32(x))
x_test['Kroutící moment'] = x_test['Kroutící moment'].apply(lambda x: np.float32(x))
x_test['Rychlost otáček'] = x_test['Rychlost otáček'].apply(lambda x: np.int32(x))
x_test['Opotřebení nástroje'] = x_test['Opotřebení nástroje'].apply(lambda x: np.int32(x))
x_test['Porucha'] = x_test['Porucha'].apply(lambda x: np.int32(x))

print(x_test.info())

#gs = HillClimbSearch(df)
#ml_model = gs.estimate(scoring_method = BDeuScore(x_test), max_iter = 5)

#plot(ml_model, node_sizes)


# COMMAND ----------

# MAGIC %run ./Plotter

# COMMAND ----------

gs = HillClimbSearch(df)
ml_model = gs.estimate(scoring_method = BicScore(x_test), max_iter = 5)

plot(ml_model, node_sizes)


# COMMAND ----------

from pgmpy.base import DAG
import networkx as nx

g = DAG()
dag = DAG([('Rychlost otáček', 'Opotřebení nástroje'), ('Rychlost otáček', 'Teplota vzduchu'), ('Rychlost otáček', 'Porucha'), ('Kroutící moment', 'Rychlost otáček'), ('Kroutící moment', 'Opotřebení nástroje'), ('Kroutící moment', 'Teplota vzduchu'), ('Opotřebení nástroje', 'Teplota vzduchu'), ('Opotřebení nástroje', 'Porucha')])
#dag = dag.to_daft(node_pos="circular")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2. Omezovací PC algoritmus

# COMMAND ----------

pc = PC(data=df)
pc_model_chi = pc.estimate(ci_test = 'chi_square', significance_level=0.05)

plot(pc_model_chi, node_sizes)

# COMMAND ----------

pc = PC(data=df)
pc_model_chi = pc.estimate(ci_test = 'chi_square', significance_level=0.15)

plot(pc_model_chi, node_sizes)

# COMMAND ----------

pc = PC(data=df)
pc_model_peaerson = pc.estimate(ci_test = 'pearsonr', significance_level=0.05)

plot(pc_model_peaerson, node_sizes)

# COMMAND ----------

pc = PC(data=df)
pc_model_peaerson = pc.estimate(ci_test = 'pearsonr', significance_level=0.1)

plot(pc_model_peaerson, node_sizes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Učení parametrů
# MAGIC 
# MAGIC 1. Kompletní data
# MAGIC     * a. MLE
# MAGIC     * b. BE
# MAGIC 2. Nekompletní data
# MAGIC     * a. EM algoritmus
# MAGIC     * b. SEM algoritmus
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Kompletní data

# COMMAND ----------

from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

mle_ml_model = MaximumLikelihoodEstimator(BayesianNetwork(ml_model), x_test)
mle_model_peaerson = MaximumLikelihoodEstimator(BayesianNetwork(pc_model_peaerson), x_test)


be_ml_model = BayesianNetwork(ml_model)
be_model_peaerson = BayesianNetwork(pc_model_peaerson)

be_ml = BayesianEstimator(be_ml_model, x_test)
be_pearson = BayesianEstimator(be_model_peaerson, x_test)

# COMMAND ----------

# MAGIC %run ./Get_CPD

# COMMAND ----------

df['Rychlost otáček'].unique()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Nekompletní data
# MAGIC 
# MAGIC #### Tvorba chybějících dat

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Parametry

# COMMAND ----------



# COMMAND ----------

from pgmpy.estimators import ExpectationMaximization as EM

estimator_k2 = EM(be_ml_model, x_test)
estimator_pearson = EM(be_model_peaerson, x_test)


# COMMAND ----------

EM_ml_est = estimator_k2.get_parameters(max_iter = 10)

# COMMAND ----------

EM_pearson_est = estimator_pearson.get_parameters(max_iter = 10)

# COMMAND ----------

print(EM_ml_est[4])

# COMMAND ----------

print(EM_pearson_est[1])

# COMMAND ----------


import pandas as pd

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df.drop(columns = ['Unnamed: 0', 'Type', 'Process temperature [K]'])
df.columns = ['Teplota vzduchu', 'Rychlost otáček', 'Kroutící moment', 'Opotřebení nástroje', 'Porucha']

df[df['Porucha'] == 1].reset_index(drop = True) 

from sklearn.model_selection import train_test_split

_, x_test, _, y_test = train_test_split(
                                                     df.loc[:, df.columns != 'b'], 
                                                     df['Porucha'], 
                                                     test_size=0.9, 
                                                     random_state=42
                                                   )

x_test['Porucha'] = y_test
x_test

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import ExpectationMaximization as EM

model = BayesianNetwork([("Porucha", "Rychlost otáček"), ("Kroutící moment", "Porucha"), ("Porucha", "Opotřebení nástroje"), ("Porucha", "Teplota vzduchu")])

estimator = EM(model, x_test)

est = estimator.get_parameters(max_iter = 1)

print_full(est[0])

# COMMAND ----------

from pgmpy.factors.discrete.CPD import TabularCPD

def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup
    
