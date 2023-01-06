# Databricks notebook source
# MAGIC %pip install pgmpy

# COMMAND ----------

# MAGIC %pip install sklearn

# COMMAND ----------

# MAGIC %pip install networkx

# COMMAND ----------

import warnings
from itertools import product, chain

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.global_vars import SHOW_PROGRESS


class EMM(ParameterEstimator):
    def __init__(self, model, data, **kwargs):

        if not isinstance(model, BayesianNetwork):
            raise NotImplementedError(
                "Expectation Maximization is only implemented for BayesianNetwork"
            )

        super(EMM, self).__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

    def _get_likelihood(self, datapoint):

        likelihood = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cpd in self.model_copy.cpds:
                scope = set(cpd.scope())
                likelihood *= cpd.get_value(
                    **{key: value for key, value in datapoint.items() if key in scope}
                )
        return likelihood

    def _compute_weights(self, latent_card):

        cache = []

        data_unique = self.data.drop_duplicates()
        n_counts = self.data.groupby(list(self.data.columns)).size().to_dict()

        for i in range(data_unique.shape[0]):
            v = list(product(*[range(card) for card in latent_card.values()]))
            latent_combinations = np.array(v, dtype=int)
            df = data_unique.iloc[[i] * latent_combinations.shape[0]].reset_index(
                drop=True
            )
            for index, latent_var in enumerate(latent_card.keys()):
                df[latent_var] = latent_combinations[:, index]

            weights = df.apply(lambda t: self._get_likelihood(dict(t)), axis=1)
            df["_weight"] = (weights / weights.sum()) * n_counts[
                tuple(data_unique.iloc[i])
            ]
            cache.append(df)

        return pd.concat(cache, copy=False), weights.sum()

    def _is_converged(self, new_cpds, atol=1e-08):
        """
        Checks if the values of `new_cpds` is within tolerance limits of current
        model cpds.
        """
        for cpd in new_cpds:
            print(type(cpd))
            if not cpd.__eq__(self.model_copy.get_cpds(node=cpd.scope()[0]), atol=atol):
                return False
        return True

    def get_parameters(
        self,
        latent_card=None,
        max_iter=100,
        atol=1e-08,
        n_jobs=-1,
        seed=None,
        show_progress=True,
    ):

        # Step 1: Parameter checks
        if latent_card is None:
            latent_card = {var: 2 for var in self.model_copy.latents}

        # Step 2: Create structures/variables to be used later.
        n_states_dict = {key: len(value) for key, value in self.state_names.items()}
        n_states_dict.update(latent_card)
        for var in self.model_copy.latents:
            self.state_names[var] = list(range(n_states_dict[var]))

        # Step 3: Initialize random CPDs if starting values aren't provided.
        if seed is not None:
            np.random.seed(seed)

        cpds = []
        for node in self.model_copy.nodes():
            parents = list(self.model_copy.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality={
                        var: n_states_dict[var] for var in chain([node], parents)
                    },
                    state_names={
                        var: self.state_names[var] for var in chain([node], parents)
                    },
                )
            )

        self.model_copy.add_cpds(*cpds)

        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(total=max_iter)

        # Step 4: Run the EM algorithm.
        iter_counter = 0
        for i in range(max_iter):
            iter_counter += 1
            # Step 4.1: E-step: Expands the dataset and computes the likelihood of each
            #           possible state of latent variables.
            weighted_data, log_lik = self._compute_weights(latent_card)
            # Step 4.2: M-step: Uses the weights of the dataset to do a weighted MLE.
            new_cpds = MaximumLikelihoodEstimator(
                self.model_copy, weighted_data
            ).get_parameters(n_jobs=n_jobs, weighted=True)

            # Step 4.3: Check of convergence and max_iter
            if self._is_converged(new_cpds, atol=atol):
                if show_progress and SHOW_PROGRESS:
                    pbar.close()
                return {"cpds": new_cpds,
                        "iter": 'converged',
                        "LL": log_lik
                
                }

            else:
                self.model_copy.cpds = new_cpds
                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)
        

        return {    "cpds": cpds,
                    "iter": "non-converged",
                    "LL": log_lik
        }


# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder

from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch

import pandas as pd
import numpy as np


from pgmpy.estimators import HillClimbSearch, BicScore, PC

import networkx as nx
import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure


import numpy as np
from scipy.special import gammaln
from math import lgamma, log

from pgmpy.estimators import StructureScore


# COMMAND ----------


class MLScore(StructureScore):
    def local_score(self, variable, parents):

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents)
        sample_size = len(self.data)
        num_parents_states = float(state_counts.shape[1])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts 

        return np.sum(log_likelihoods)


# COMMAND ----------

def Search_greedily(df, method):
  gs = HillClimbSearch(df)
  plot(gs.estimate(scoring_method=BicScore(df)), node_sizes),
  
  return gs.estimate(scoring_method=BicScore(df))

# COMMAND ----------

def sample_df(df):

  df = df.sort_values(by = ['Machine failure'], ascending = True)
  df_2 = df.sort_values(by = ['Machine failure'], ascending = False)

  df_2 = df_2.head(10)
  df = df.head(1000 )

  return pd.concat([df, df_2])


# COMMAND ----------

def add_cpd(params):
    "After initial layer, adds Conditional Probability distribution to Bayesian Network model"

    for param in params:
        bn.add_cpds(param)

    return bn

# COMMAND ----------

def check_runs(layer):
    "Checks and pushes layers"
    #print('Layer: \n')
    
    #print(layer)
    
    #print('--'*20)
    
    new_layer = []
    # check tolerance
    for run in layer:
      print('is converged: ')
      print(run.get('iter') == 'converged')
      print('---' * 20)
      
      
      
      if run.get('iter') == 'converged':
          new_layer.append(run)
      elif len(new_layer) > 0:
        for r in new_layer:
          if run.get('LL') > r.get('LL'):
              new_layer.pop(r)
      else:
          pass
    
    
    print('New_layer: \n')
    print(new_layer)
    print('--' * 20)
    
    
    return new_layer

# COMMAND ----------

def get_next_layer(layer, model):
    "Gets next age layer in ALEM algorithm"

    lst = []
    for run in layer:
        next_layer = BayesianNetwork(model, df_2)                   \
                    .get_parameters(max_iter = 1)                   \
                    .add_cpds(run.get('cpds')
            )                                                       #prida dalsi beh
        
        lst.append(next_layer)

    return lst

# COMMAND ----------

def are_all_converged(lst):
    from itertools import groupby
    convergence_lst = []

    def are_all_equal(convergence_lst):
      "Checks for equality in list of dicts by dict key"

      convergence_info = groupby(convergence_lst)

      return next(convergence_info, True) and not next(convergence_info, False)
    
    for run in lst:
      convergence_lst.append(run.get('iteration'))

    return are_all_equal(lst)

# COMMAND ----------

def Get_best_run(next_layer):
  "Sorts list of dicts by key in dict"

  return sorted(next_layer, key=lambda d: d['LL'])

# COMMAND ----------

def ALEM(initial_values, total_iterations, em_iter, layers, model, df_2):

    bn = BayesianNetwork(model)
    layer_lst = []
    #inicializuj prvni vrstvou
    for i in initial_values:
        layer_lst.append(ExpectationMaximization(bn, df_2).get_parameters(max_iter = em_iter, seed = i))

    i = 1
    
    print('layer_after_first_run: ')
    print(layer_lst)
    print('--')
    
    while i < total_iterations:
      # ALEM algorithm
      if i == 1:                                            #koukni na dalsi vrstvu
        if are_all_converged(layer_lst) == True:            #Pokud jiz pri prvnim behu doslo ke konvergenci
          next_layer = layer_lst                            #Pro obecny return potreba prejmenovat list
          break
        
        new_layer = check_runs(layer_lst)                   #Zkontroluj podminky pro terminaci algoritmu
        next_layer = get_next_layer(new_layer, bn)          #Urci dalsi vrstvu
        print(next_layer)
        i+=1

      else:
        current_layer = next_layer                          #Prejmenuj predeslou dalsi na novou nynejsi
        next_layer = check_runs(current_layer)              #Urci dalsi vrstvu
        next_layer = get_next_layer(next_layer, bn)
        i+=1       

      if are_all_converged(next_layer) == True:             #ALEM predopklada posun nezkonvergovanych behu do dalsi vrstvy
                                                            #Pokud sou vsechny zkonvergovane posouvaly by se automaticky az do posledni vrstvy
        break


    
    return Get_best_run(next_layer)                         #Vrati datovou strukturu "dictionary", ktera v sobe nese CPD, LL, informaci o konvergenci



            


# COMMAND ----------

from random import seed
from random import randint

def create_missing_values(df):
  
  col_lst = ['variable', 'year', 'data_value']
  missing_values = []

  for name in col_lst:
    for index in np.random.randint(0, len(df), 150):
      df[name].loc[index] = np.nan
  
  return df


# COMMAND ----------


enc = OrdinalEncoder()
df_2 = pd.read_csv('/dbfs/mnt/pbi/Bots/PeopleBook/green.csv')

for i in df_2.columns:
    if type(df_2[i][0]) == str:
        df_2[i] = enc.fit_transform(df_2[[i]])

df_2 = df_2.drop(columns = ['nzsioc'])



df_2 = df_2.head(1500)
df_2['data_value'] = pd.qcut(df_2['data_value'], q=15, duplicates = 'raise')


initial_values = 5
total_iterations = 100
layers = 5
em_iter = 2

gs = HillClimbSearch(df_2.dropna())
ml_model = gs.estimate(scoring_method = MLScore(df_2.dropna()), max_iter = 10)

df_2 = create_missing_values(df_2)
df_2.info()

# COMMAND ----------

def ALEM(init_runs, total_iterations, em_iter, layers, model, df_2):

    bn = BayesianNetwork(model)
    layer_lst = []
    #inicializuj prvni vrstvou
    for seed in range(1, init_runs + 1):
      print(seed)
      run = EMM(
          bn, 
          df_2
        ).get_parameters(
          max_iter = em_iter
         , seed = 2
        )
      layer_lst.append(run)

    i = 1
    
    print('layer_after_first_run: ')
    print(layer_lst)
    print('--')
    
    while i < total_iterations:
      # ALEM algorithm
      if i == 1:                                            #koukni na dalsi vrstvu
        if are_all_converged(layer_lst) == True:            #Pokud jiz pri prvnim behu doslo ke konvergenci
          next_layer = layer_lst                            #Pro obecny return potreba prejmenovat list
          break
        
        new_layer = check_runs(layer_lst)                   #Zkontroluj podminky pro terminaci algoritmu
        next_layer = get_next_layer(new_layer, bn)          #Urci dalsi vrstvu
        print(next_layer)
        i+=1

      else:
        current_layer = next_layer                          #Prejmenuj predeslou dalsi na novou nynejsi
        next_layer = check_runs(current_layer)              #Urci dalsi vrstvu
        next_layer = get_next_layer(next_layer, bn)
        i+=1       

      if are_all_converged(next_layer) == True:             #ALEM predopklada posun nezkonvergovanych behu do dalsi vrstvy
                                                            #Pokud sou vsechny zkonvergovane posouvaly by se automaticky az do posledni vrstvy
        break


    
    return Get_best_run(next_layer)                         #Vrati datovou strukturu "dictionary", ktera v sobe nese CPD, LL, informaci o konvergenci



a = ALEM(initial_values, total_iterations, em_iter, layers, ml_model, df_2)

# COMMAND ----------

bn = BayesianNetwork(ml_model)

#for seed in range(1, init_runs + 1):
#  print(seed)
run = EMM(
    bn, 
    df_2
  ).get_parameters(
    #max_iter = em_iter
    #, seed = 2
  )
layer_lst.append(run)