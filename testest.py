# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# COMMAND ----------

from typing import Optional, Tuple
import numpy as np

class ImpPlot:
  def __init__(self, SMALL_SIZE = None, BIG_SIZE = None):
    self.SMALL_SIZE = SMALL_SIZE
    self.BIG_SIZE = BIG_SIZE
  
  @staticmethod
  def _config_text(SMALL_SIZE: Optional[int] = None, BIG_SIZE: Optional[int] = None):  
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIG_SIZE)
    
  @staticmethod
  def filter_error_iteration(err, iter, iter_to_plot: int):
    
    return [x for x, y in list(zip(err, iter)) if y == iter_to_plot]
  
  @staticmethod
  def save_figure(
                  distribution: Optional[str] = None,
                  tol: Optional[str] = None,
                  name: Optional[str] = None,
                  type_plot: Optional[str] = None
                ):
  
    if (method and tol and name) is not None:
      plt.savefig(f'/dbfs/mnt/pbi/Bots/T_budget/{type_plot}_{name}_{distribution}_{tol}.png')
      plt.savefig(f'/dbfs/mnt/pbi/Bots/T_budget/{type_plot}_{name}_{distribution}_{tol}.png')
    elif name is not None:
      plt.savefig(f'/dbfs/mnt/pbi/Bots/T_budget/hist__{name}.png')

      
  def histplot(
                self, 
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
    #plt.show()
  
  def err_plot(self,   
                  err: np.array,
                  iter: np.array,
                  y_color: str, 
                  y_label: str,
                  iter_to_plot: int,
                  units: Optional[str],
                  y_lim: Optional[list] = None,
                  distribution: Optional[str] = None,
                  tol: Optional[str] = None,
                  name: Optional[str] = None
              ):
    
    y = self.filter_error_iteration(err, iter, iter_to_plot)
    if units is None:
      units = '[-]'
    #Figure grid setup
    fig, ax = plt.subplots(figsize = (20, 10))
    
    if (self.SMALL_SIZE is not None) and (self.SMALL_SIZE is not None):
      self._config_text(SMALL_SIZE = self.SMALL_SIZE, BIG_SIZE = self.BIG_SIZE)
    
    #grid max-min
    plt.xlim([0, len(y) + 1])
    
    if y_lim is not None:
      plt.ylim(y_lim)
    
    #Plot baselines
    plt.axhline(y = 0, color = 'k', linestyle = '-')
    plt.axhline(y = 0.01, color = 'r', linestyle = ':')
    plt.axhline(y = -0.01, color = 'r', linestyle = ':')
    
    #plot legend
    ax.set_xlabel(f"{iter_to_plot}-tá iterace {units}")
    ax.set_ylabel(f"Velikost chyby {units}")
    ax.set_title(f'Velikost chyby v závislosti na iteraci pro {iter_to_plot}-tý chybějící bod')
    
    #plotting
    plt.plot(y, y_color, label = y_label)
    plt.legend()
    
    self.save_figure(
                  distribution = distribution,
                  tol = tol,
                  name = name,
                  type_plot = "error"
                )
    
    return plt.show()

  
  def imp_plot( self,
                y: np.array,
                y2: np.array,
                y_color: str, 
                y_label: str,
                y2_color: str, 
                y2_label: str,
                units: Optional[str] = None,
                legend_position: Optional[str] = None,
                distribution: Optional[str] = None,
                tol: Optional[str] = None,
                name: Optional[str] = None,
                type_name: Optional[str] = None
              ):
    
    x = np.arange(1, len(y) + 1)
    if units is None:
      units = '[-]'
    #Figure grid setup
    fig, ax = plt.subplots(figsize = (20, 10))
    
    #grid max-min
    lower_lim = min(np.minimum(y, y2)) - 0.5
    upper_lim = max(np.maximum(y, y2)) + 0.5
    
    plt.xlim([0, len(y) + 1])
    plt.ylim([lower_lim, upper_lim])
    
    #Plot baselines
    plt.axhline(y = 0, color = 'k', linestyle = '')
    
    #Plot legend
    ax.set_xlabel("I-tý nezmáný bod [-]")
    ax.set_ylabel(f"Hodnota bodu {units}")
    ax.set_title('Rozdíl skutečné a odhadnuté hodnoty')
    
    #Plot lines
    plt.plot(x, y, y_color, label = y_label)
    plt.plot(x, y2, y2_color, label = y2_label)
    
    plt.legend()
    
    self.save_figure(
                  distribution = distribution,
                  tol = tol,
                  name = type_name,
                  type_plot = "impute"
                )
    
    
    return plt.show()


    


# COMMAND ----------

col = 'Target'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

#################################################################################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.03, random_state=42)
df = x_test

# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = x_test['Target'], 
                bins = 50, 
                x_name = 'Porucha',
                title = 'Porucha',
                figsize = (16, 8)
            )