# Databricks notebook source
import networkx as nx
import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure


# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
#from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot(model, node_sizes, from_list: Optional[str] = None):
  font = 48
  plt.rc('font', size=font)
  plt.rc('axes', titlesize=font)
  plt.rc('axes', labelsize=font)
  plt.rc('legend', fontsize=font)
  plt.rc('figure', titlesize=font)
    
  if from_list is None:
    plt.figure(figsize=(8, 6), dpi=100)  

    nx.draw(model, with_labels=True, node_size = node_sizes)

    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
  else:
    plt.figure(figsize=(8, 6), dpi=100)  
  
    nx.draw(model, with_labels=True, node_size = node_sizes)

    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])

  return plt.show()

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Kroutící moment', 'Porucha'), ('Porucha', 'Rychlost otáček'), ('Porucha', 'Opotřebení nástroje'), ('Porucha', 'Teplota vzduchu')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Teplota vzduchu', 'Kroutící moment'), ('Rychlost otáček', 'Kroutící moment'), ('Rychlost otáček', 'Teplota vzduchu')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Opotřebení nástroje', 'Porucha'), ('Teplota vzduchu', 'Rychlost otáček')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Opotřebení nástroje', 'Porucha'), ('Teplota vzduchu', 'Rychlost otáček')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Opotřebení nástroje', 'Porucha'), ('Teplota vzduchu', 'Rychlost otáček')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(BayesianNetwork([('Teplota vzduchu', 'Kroutící moment'), ('Rychlost otáček', 'Kroutící moment'), ('Rychlost otáček', 'Teplota vzduchu')]), 5000)

# COMMAND ----------

from pgmpy.models import BayesianNetwork
plot(
  BayesianNetwork(
    [('Teplota vzduchu', 'Kroutící moment'), ('Rychlost otáček', 'Kroutící moment'), ('Rychlost otáček', 'Teplota vzduchu')]), 
  5000
)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from pgmpy.models import BayesianNetwork
import networkx as nx
plot(BayesianNetwork(([])), 5000)

plot(BayesianNetwork([('Teplota vzduchu', 'Otáčky za minutu'), ('Opotřebení nástroje', 'Porucha')]), 5000)

plot(BayesianNetwork([('Teplota vzduchu', 'Otáčky za minutu'), ('Kroutící moment', 'Otáčky za minutu'), ('Kroutící moment', 'Teplota vzduchu')]), 5000)

# COMMAND ----------

