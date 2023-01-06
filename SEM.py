# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Structural Expecation Maximization Algorithm

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %run ./ImputationClass

# COMMAND ----------

# MAGIC %run ./ML_score

# COMMAND ----------


df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df.drop(columns = ['Unnamed: 0', 'Type', 'Process temperature [K]'])
df.columns = ['Teplota vzduchu', 'Rychlost otáček', 'Kroutící moment', 'Opotřebení nástroje', 'Porucha']

df[df['Porucha'] == 1].reset_index(drop = True) 
_, x_test, _, y_test = train_test_split(
                                           df.loc[:, df.columns != 'b'], 
                                           df['Porucha'], 
                                           test_size=0.03, 
                                           random_state=42
                                         )


# COMMAND ----------

x_test['Teplota vzduchu'] = x_test['Teplota vzduchu'].apply(lambda x: np.float32(x))
x_test['Kroutící moment'] = x_test['Kroutící moment'].apply(lambda x: np.float32(x))
x_test['Rychlost otáček'] = x_test['Rychlost otáček'].apply(lambda x: np.int32(x))
x_test['Opotřebení nástroje'] = x_test['Opotřebení nástroje'].apply(lambda x: np.int32(x))
x_test['Porucha'] = x_test['Porucha'].apply(lambda x: np.int32(x))

# COMMAND ----------

'''
from pgmpy.estimators import ExpectationMaximization as EM

estimator_k2 = EM(be_ml_model, df_missing)
estimator_pearson = EM(be_model_peaerson, df_missing)
'''

# COMMAND ----------

from typing import Optional, List, Tuple

class SEM(EM_impute):
  def __init__(
                self, 
                max_iter_structure: int, 
                max_iter_params: int, 
                variable_list: List[str],
                array = None, 
                distr = None
            ):
    
    self.array = array,
    self.max_iter_structure = max_iter_structure,
    self.max_iter_params = max_iter_params,
    self.variable_list = variable_list

    super().__init__(distr, array)
    
  def _impute_missing_variables(
                                  self,
                                  missing_values_range: Tuple[int, int],
                                  distribution,
                                  data
                          ):
    
    for variable in self.variable_list:
      array = data[variable]
      imputed_variable = self.em_imputation(distribution)
      
      data[missing_values_range[0]:missing_values_range[1]] = imputed_variable
      
    return data
  
  def _create_missing_values(self, missing_values_range, data):
    df = data
    df = pd.DataFrame(df)
    print(df)
    print(type(df))
    
    for variable in self.variable_list:
      print(variable)
      #df[variable][0:50] = np.nan
    
    return data
  
  def _structure_learn(self, method: str, data):
    
    if self.method == "HillClimb_aic":
      gs = HillClimbSearch(data)
      return gs.estimate(scoring_method = BicScore(data), max_iter = self.max_iter_structure)
    elif self.method == "HillClimb_bic":
      gs = HillClimbSearch(data)
      return gs.estimate(scoring_method = MLScore(data), max_iter = self.max_iter_structure)
    elif self.method == "PC-chi":
      pc = PC(data=data)
      return pc.estimate(ci_test = 'chi_square', significance_level=0.05)
    elif self.method == "Exhaustive":
      gs = ExhaustiveSearch(data, scoring_method=MLScore(data))
      return gs.estimate()
    elif self.method == "PC-pearson": 
      pc = PC(data=data)
      return pc.estimate(ci_test = 'pearsonr', significance_level=0.05)
    else:
      raise ValueError("Nebyla zadana metoda uceni. Prosim zadej metodu uceni!")
  
  def _learn_parameters(self, model, data):
    
    estimator = EM(model, data)
    
    return estimator.get_parameters(max_iter = max_iter)
  
  def SEM_learn(
                  self,
                  missing_values_range: Tuple[int, int], 
                  method: str, 
                  distr: List[str]
              ):
    
        
    #print(f"type data SEM_learn {self.data.info()}")
    print(f"type data SEM_learn {type(self.data)}")
    
    self.distr = distr
      
    ### E - Step
    
    self.data = self._create_missing_values(missing_values_range)
    self.data = self._impute_missing_variables(missing_values_range, distribution = distr)
    
    ### M - Step
    model = self._structure_learn(method = method)
    
    params = self._learn_parameters(model)
    
    
    return params, model

######################################################################################################################
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df = df.drop(columns = ['Unnamed: 0', 'Type', 'Process temperature [K]'])
df.columns = ['Teplota vzduchu', 'Rychlost otáček', 'Kroutící moment', 'Opotřebení nástroje', 'Porucha']

_, x_test, _, y_test = train_test_split(
                                           df.loc[:, df.columns != 'b'], 
                                           df['Porucha'], 
                                           test_size=0.03, 
                                           random_state=42
                                         )

x_test['Porucha'] = y_test

################################################################################################################################


sem = SEM(
          max_iter_structure = 2, 
          max_iter_params = 2,
          variable_list = ['Teplota vzduchu', 'Kroutící moment', 'Rychlost otáček'],
          data = x_test
        )

params, model = sem.SEM_learn(
                              missing_values_range = (0, 50), 
                              method = "HillClimb_aic", 
                              distr = ["normal", "normal", "normal"]
                            )
  
  

# COMMAND ----------

missing_values_range = (0, 50)

missing_values_range[0]

x_test['Kroutící moment'][0:50] = np.nan