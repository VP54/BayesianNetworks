# Databricks notebook source
#pip install scipy

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# COMMAND ----------

from typing import Tuple

class DistributionInfo:
  def __init__(self, vals):
    self.vals = vals
  
  def _mean(self):
    vals_lst = self.vals[~np.isnan(self.vals)]
    
    return sum(self.vals) / len(self.vals)
  
  def _moments(self, moment):
    
    vals_mean = self._mean()
    prod = [
      (val - vals_mean) ** moment 
      for val 
      in self.vals]
      
    return sum(prod) / len(self.vals)
  
  def _skew(self):
    
    len_vals = len(self.vals)
    moment_ratio = self._moments(3) / self._moments(2) ** 1.5

    return (len_vals * (len_vals - 1)) ** 0.5 / ((len_vals - 2)) * (moment_ratio)
  
  def _std_dev(self):
      mean = self._mean()
      citatel = [(val - mean) ** 2 for val in self.vals ]

      return (sum(citatel) / (len(self.vals) - 1)) ** 0.5
    
  def _kurtosis(self):
    
    return self._moments(4) / self._moments(2) ** 2
  
  def get_kurtosis_skewness(self):# -> Tuple[float, float]:
  
    return self._kurtosis(), self._skew()
  
  

# COMMAND ----------

class EM_impute(DistributionInfo):
    from typing import Optional, Tuple
    from scipy.stats import skewnorm
    
    def __init__(self, array, vals = None):
        self.array = array
        super().__init__(vals)

    def _get_from_normal(self):
        array = self.array[~np.isnan(self.array)]

        return np.random.normal(
            array.mean(), 
            array.std()
            )
        
    def _get_from_skewed_normal(self):
      
      kurtosis, skewness = self.get_kurtosis_skewness()
      std = self._std_dev()
      mean = self._mean()
      
      print(f"Sikmost rozdeleni: {skewness} \t a spicatost: {kurtosis} \n")
      
      return skewnorm.rvs(loc = mean, scale = std, size=1)
    
    def _get_from_uniform(
        self,
        min: float,
        max: float
    ):
    
      return np.random.uniform(min, max)
    
    def _impute_skewnorm(
                self,
                nan_array: np.array, 
                nan_index: np.array, 
                tol: Optional[float] = None
                ):
      
        val_lst = []
        index_lst = []
        prev  = 100000
        prvek = 0
        counter = 0

        for _ in nan_array:
            prvek += 1
            val = self._get_from_normal()
            delta = np.abs(val - prev) / prev
            val_lst.append(delta)
            index_lst.append(prvek)
            prev = val
          
            if tol:
                while np.abs(delta) > tol:
                  val = self._get_from_skewed_normal()
                  delta = np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)

                self.array[nan_index[prvek - 1]] = val
            else:
                while np.abs(delta) > 0.01:
                  val = self._get_from_skewed_normal()
                  delta =  np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)

                self.array[nan_index[prvek - 1]] = val

        return self.array, val_lst, index_lst

    def _impute_normal(
                self,
                nan_array: np.array, 
                nan_index: np.array, 
                tol: Optional[float] = None
                ):
      
        val_lst = []
        index_lst = []
        prev  = 100000
        prvek = 0
        counter = 0

        for _ in nan_array:
            prvek += 1
            val = self._get_from_normal()
            delta = np.abs(val - prev) / prev
            val_lst.append(delta)
            index_lst.append(prvek)
            prev = val
          
            if tol:
                while np.abs(delta) > tol:
                  val = self._get_from_normal()
                  delta = np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)

                self.array[nan_index[prvek - 1]] = val
            else:
                while np.abs(delta) > 0.01:
                  val = self._get_from_normal()
                  delta =  np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)

                self.array[nan_index[prvek - 1]] = val

        return self.array, val_lst, index_lst
      
    def _impute_uniform(
                self,
                nan_array: np.array, 
                nan_index: np.array, 
                tol: Optional[float] = None
                ):
        
        val_lst = []
        index_lst = []
        prev  = 100000
        prvek = 0

        for _ in nan_array:
            prvek += 1
            val = self._get_from_normal()
            delta = np.abs(val - prev) / prev
            val_lst.append(delta)
            index_lst.append(prvek)
            prev = val
          
            if tol:
                while np.abs(delta) > tol:
                  val = self._get_from_uniform()
                  delta = np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)               

                self.array[nan_index[prvek - 1]] = val
            else:
                while np.abs(delta) > 0.01:
                  val = self._get_from_uniform()
                  delta =  np.abs(val - prev) / prev
                  prev = val
                  val_lst.append(delta)
                  index_lst.append(prvek)

                self.array[nan_index[prvek - 1]] = val

        return self.array, val_lst, index_lst

    def em_imputation(self, **params) -> np.array:
      
      for k, v in params.items():
        print(f"{k} \t\t --> \t\t {v}")

      self.array[5:15] = np.nan

      nan_array = self.array[np.isnan(self.array)]
      nan_index = np.argwhere(np.isnan(self.array))[:,0]

      if params["distribution"] == "normal":
          imputed_array, val_lst, index_lst = self._impute_normal(nan_array, nan_index)
      elif params["distribution"] == "dirichlet":
          imputed_array, val_lst, index_lst = self._impute_dirichlet(nan_array, nan_index, params["m"], params["n"], params["size"])
      elif params["distribution"] == "skew_normal":
          imputed_array, val_lst, index_lst = self._impute_normal(nan_array, nan_index)
      else:
        raise ValueError("Zadej vsechny hodnoty prosim.")

      return imputed_array, val_lst, index_lst

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
    ax.set_ylabel(f"Velikost relativního rozdílu i-té a předchozí iterace")
    ax.set_title(f'Velikost relativního rozdílu i-té a předchozí iterace pro {iter_to_plot}-tý chybějící bod')
    
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

class ImpError:
  def __init__(self, actual: np.array, imputed: np.array):
    self.actual = actual
    self.imputed = imputed
    
   
  def _calculate_error_point(self, name, error_lst = []):
    for actual, imputed in list(zip(self.actual, self.imputed)):
      err = np.abs((imputed - actual)/actual)
      error_lst.append(err)

      #print(f"Actual: {actual} \t Imputed: {imputed} \t err: {err} \n")
      np.savetxt(f"/dbfs/mnt/pbi/Bots/err_{name}.csv", error_lst, delimiter=",")
      
      
      
    return sum(error_lst) / len(self.imputed)
  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Globals

# COMMAND ----------

EM_SKUTECNA = "Skutečná hodnota"
EM_ODHADNUTA = "Hodnota odhanutá EM algoritmem"
CHYBA_ODHADU = "Chyba odhadu chybějící hodnoty"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Air temperature

# COMMAND ----------


#############################################
col = 'Air temperature [K]'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

#################################################################################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.99, random_state=42)
df = x_test
#################################################################################################################################

copy_arr_air_temp = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_normal_air_temp, error_lst_normal_air_temp, index_lst_normal_air_temp = impute.em_imputation(distribution = "normal", tol = 0.001)

copy_arr_air_temp = copy_arr_air_temp.to_numpy()

error_normal = ImpError(
  actual = copy_arr_air_temp[0:50],
  imputed = pred_normal_air_temp[0:50]
)

np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_normal_air.csv", pred_normal_air_temp[0:50], delimiter=",")
np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/orig_normal_air.csv", copy_arr_air_temp[0:50], delimiter=",")

error_normal._calculate_error_point(name = col)



# COMMAND ----------

ITER_TO_PLOT = 25

# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )

ITER_TO_PLOT = 35

pplot.err_plot(
                error_lst_normal_air_temp, 
                index_lst_normal_air_temp, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.15], 
                units = '[K]', 
                distribution = "normal", 
                tol = '0.01', 
                name = 'teplota_vzduchu'
            )



# COMMAND ----------

pplot.err_plot(
                error_lst_normal_air_temp, 
                index_lst_normal_air_temp, 
                'o', CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.05, 0.05],
                units = "[K]",
                distribution = "normal", 
                tol = '0.01', 
                name = 'teplota_vzduchu'
            )

# COMMAND ----------

pplot.err_plot(
                error_lst_normal_air_temp, 
                index_lst_normal_air_temp, 
                'o', CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0,1, 0.15], 
                units = '[K]', 
                distribution = "normal", 
                tol = '0.01', 
                name = 'teplota_vzduchu'
            )

# COMMAND ----------

x_test.head(50)

# COMMAND ----------

pplot.imp_plot(
               y = copy_arr_air_temp[0:50],
               y2 = pred_normal_air_temp[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA
              )

# COMMAND ----------

##########################################################################################################################################################################################################################################

# COMMAND ----------

# MAGIC %md
# MAGIC # Kroutici moment

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

col = 'Torque [Nm]'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

###########################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.99, random_state=42)
df = x_test
#################################################################

copy_arr_torque = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_normal_torque, error_lst_normal_torque, index_lst_normal_torque = impute.em_imputation(distribution = "normal", tol = 0.01)

copy_arr_torque = copy_arr_torque.to_numpy()

error_normal = ImpError(
  actual = copy_arr_torque[0:50],
  imputed = pred_normal_torque[0:50]
)

np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_normal_torque.csv", pred_normal_torque[0:50], delimiter=",")
np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/orig_normal_torque.csv", copy_arr_torque[0:50], delimiter=",")

error_normal._calculate_error_point(name = col)

# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )
pplot.err_plot(
                error_lst_normal_torque, 
                index_lst_normal_torque, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.2, 1], 
                units = '[Nm]', 
                distribution = "normal", 
                tol = '0.01', 
                name = 'kroutici_moment0'
            )


# COMMAND ----------

pplot.err_plot(
                error_lst_normal_torque, 
                index_lst_normal_torque, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, y_lim = [-0.1, 0.35], 
                units = '[Nm]',  
                distribution = "normal", 
                tol = '0.01', 
                name = 'kroutici_moment1'
            )


# COMMAND ----------

pplot.err_plot(
                error_lst_normal_torque, 
                index_lst_normal_torque, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, y_lim = [-0.05, 0.15], 
                units = '[Nm]', 
                distribution = "normal", 
                tol = '0.01', 
                name = 'kroutici_moment2'
            )


# COMMAND ----------

pplot.imp_plot(
               y = copy_arr_torque[0:50],
               y2 = pred_normal_torque[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA,
               units = '[Nm]', 
               distribution = "normal", 
               tol = '0.01', 
               name = 'kroutici_moment3'
              )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Universal plot

# COMMAND ----------

'''
pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )
pplot.err_plot(error_lst_normal_torque, index_lst_normal_torque, 'o', 'Chyba odhadu chybejici hodnoty', iter_to_plot = 20, y_lim = [-1, 1])

pplot.err_plot(error_lst_normal, index_lst_normal, 'o', 'Chyba odhadu chybejici hodnoty', iter_to_plot = 20, y_lim = [-0.1, 1])

pplot.err_plot(error_lst_normal, index_lst_normal, 'o', 'Chyba odhadu chybejici hodnoty', iter_to_plot = 20, y_lim = [-0.1, 0.15])
pplot.imp_plot(
               y = copy_arr[0:50],
               y2 = pred_normal[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA
              )
p.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_normal_torque.csv", pred_normal_torque[0:50], delimiter=",")
'''

# COMMAND ----------

# MAGIC %md
# MAGIC # Otacky skewnormal rozdeleni

# COMMAND ----------

col = 'Rotational speed [rpm]'
#col = 'Torque [Nm]'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

#################################################################################33
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.99, random_state=42)
df = x_test
######################################################################################

#df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

copy_skewed_arr = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy(),  df[col][0:50])
pred_skewed_normal, error_lst_skewed_normal, index_lst_skewed_normal = impute.em_imputation(
                                                                                              tol = 0.00001, 
                                                                                              distribution = "skew_normal"
                                                                                            )

copy_skewed_arr = copy_skewed_arr.to_numpy()

error_normal = ImpError(
  actual = copy_skewed_arr[0:50],
  imputed = pred_skewed_normal[0:50]
)

np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_skewed_normal.csv", pred_skewed_normal[0:50], delimiter=",")
np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/orig_skewed_normal.csv", copy_skewed_arr[0:50], delimiter=",")


error_normal._calculate_error_point(name = col)


################################################################################################################################

# COMMAND ----------

distr = DistributionInfo(vals = df[col])
distr.get_kurtosis_skewness()  

# COMMAND ----------

l = 50
soucet = 0
for i, ii in list(zip(copy_skewed_arr[0:50], pred_skewed_normal[0:50])):
  val = (i - ii) / i
  
  soucet += val
  
soucet / l

# COMMAND ----------

len(error_lst_skewed_normal)

# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )
pplot.err_plot(
              error_lst_skewed_normal, 
              index_lst_skewed_normal, 
              'o', 
              CHYBA_ODHADU, 
              iter_to_plot = ITER_TO_PLOT, 
              y_lim = [-0.1, 0.6],
              units = "[ot/m]",
              distribution = "normal", 
              tol = '0.01', 
              name = 'otacky'
  )


# COMMAND ----------

pplot.err_plot(
              error_lst_skewed_normal, 
              index_lst_skewed_normal, 
              'o', 
              CHYBA_ODHADU, 
              iter_to_plot = ITER_TO_PLOT, 
              y_lim = [-0.1, 0.25],
              distribution = "skew_normal", 
              tol = '0.01',
              units = "[ot/min]",
              name = 'otacky1'
  )


# COMMAND ----------

pplot.err_plot(
              error_lst_skewed_normal, 
              index_lst_skewed_normal, 
              'o', 
              CHYBA_ODHADU, 
              iter_to_plot = ITER_TO_PLOT, 
              y_lim = [-0.1, 0.15], 
              distribution = "skew_normal", 
              tol = '0.01', 
              units = "[ot/min]",
              name = 'otacky2'
  )


# COMMAND ----------

pplot.imp_plot(
               y = copy_skewed_arr[0:50],
               y2 = pred_skewed_normal[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA,
               distribution = "skew_normal", 
               tol = '0.01', 
               units = "[ot/min]",
               name = 'otacky3'
              )

# COMMAND ----------

# MAGIC %md 
# MAGIC # Otacky normal tol

# COMMAND ----------

from sklearn.model_selection import train_test_split

col = 'Rotational speed [rpm]'
#col = 'Torque [Nm]'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

#################################################################################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.99, random_state=42)
df = x_test
#################################################################################################################################

copy_arr_normal_tol = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_normal_tol, error_lst_tol, index_lst_tol = impute.em_imputation(distribution = "normal", tol = 0.00001)

copy_arr_normal_tol = copy_arr_normal_tol.to_numpy()

error_normal_tol = ImpError(
  actual = copy_arr_normal_tol[0:50],
  imputed = pred_normal_tol[0:50]
)

np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_normal_tol.csv", pred_normal_tol[0:50], delimiter=",")
np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/orig_normal_tol.csv", copy_arr_normal_tol[0:50], delimiter=",")

error_normal_tol._calculate_error_point(name = col)


# COMMAND ----------

l = 50
soucet = 0
for i, ii in list(zip(copy_skewed_arr[0:50], pred_normal_tol[0:50])):
  val = (i - ii) / i
  
  soucet += val
  
soucet / l

# COMMAND ----------

pplot = ImpPlot()
pplot.err_plot(
                error_lst_tol, 
                index_lst_tol, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.15], 
                distribution = "normal", 
                tol = '0.01', 
                name = 'otacky',
                units = "[ot/min]"
            )


# COMMAND ----------

pplot = ImpPlot()

pplot.err_plot(
                error_lst_tol, 
                index_lst_tol, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.5],
                distribution = "normal", 
                tol = '0.01', 
                units = "[ot/min]",
                name = 'otacky1'
            )


# COMMAND ----------


pplot = ImpPlot()
pplot.err_plot(
                error_lst_tol, 
                index_lst_tol, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.15],
                distribution = "normal", 
                tol = '0.01', 
                units = "[ot/min]",
                name = 'otacky2'
            )

# COMMAND ----------

pplot = ImpPlot()
pplot.imp_plot(
               y = copy_arr_normal_tol[0:50],
               y2 = pred_normal_tol[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA,
               distribution = "normal", 
               tol = '0.01', 
               name = 'otacky3'
              )

# COMMAND ----------

# MAGIC %md
# MAGIC # Opotrebeni nastroje

# COMMAND ----------

col = 'Tool wear [min]'
#col = 'Torque [Nm]'

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")

#################################################################################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'b'], df['Target'], test_size=0.99, random_state=42)
df = x_test
#################################################################################################################################

copy_arr_tool_wear = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_uniform, error_lst_uniform, index_lst_uniform = impute.em_imputation(distribution = "normal", tol = 0.001)

copy_arr_tool_wear = copy_arr_tool_wear.to_numpy()

error_uniform = ImpError(
  actual = copy_arr_tool_wear[0:50],
  imputed = pred_uniform[0:50]
)

np.savetxt("/dbfs/mnt/pbi/Bots/T_budget/pred_uniform.csv", pred_uniform[0:50], delimiter=",")

error_uniform._calculate_error_point(name = col)

# COMMAND ----------


pplot.err_plot(
                error_lst_uniform, 
                index_lst_uniform, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-1, 1],
                distribution = "normal", 
                tol = '0.01', 
                units = "[ot/min]",
                name = 'opotrebeni'
            )


# COMMAND ----------


pplot.err_plot(
                error_lst_uniform, 
                index_lst_uniform, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.5],
                distribution = "normal", 
                tol = '0.01', 
                units = "[ot/min]",
                name = 'opotrebeni1'
            )


# COMMAND ----------


pplot.err_plot(
                error_lst_uniform, 
                index_lst_uniform, 
                'o', 
                CHYBA_ODHADU, 
                iter_to_plot = ITER_TO_PLOT, 
                y_lim = [-0.1, 0.5], 
                distribution = "normal", 
                tol = '0.01', 
                units = "[ot/min]",
                name = 'opotrebeni3'
            )


# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )

pplot.imp_plot(
               y = copy_arr_tool_wear[0:50],
               y2 = pred_uniform[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA, 
               distribution = "normal", 
               tol = '0.01', 
               units = "[ot/min]",
               name = "opotrebeni3"
              )

# COMMAND ----------

# MAGIC %md
# MAGIC # Histplots

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv(f"/dbfs/mnt/pbi/Bots/Bot20/predictive_maintenance.csv")
df_1000 = df.sample(500)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('fivethirtyeight')

np.random.seed(0)
train = np.random.exponential(2, size=100000)
test = df['Rotational speed [rpm]']

SAMPLE_SIZE = 10000
N_BINS = 300

# Obtain `N_BINS` equal frequency bins, in other words percentiles
step = 100 / N_BINS
test_percentiles = [
    np.percentile(test, q, axis=0)
    for q in np.arange(start=step, stop=100, step=step)
]

# Match each observation in the training set to a bin
train_bins = np.digitize(train, test_percentiles)

# Count the number of values in each training set bin
train_bin_counts = np.bincount(train_bins)

# Weight each observation in the training set based on which bin it is in
weights = 1 / np.array([train_bin_counts[x] for x in train_bins])

# Make the weights sum up to 1
weights_norm = weights / np.sum(weights)

np.random.seed(0)
sample = np.random.choice(train, size=SAMPLE_SIZE, p=weights_norm, replace=False)

# COMMAND ----------

len(test_percentiles)

# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Rotational speed [rpm]'][50:], 
                bins = 300,
                x_name = 'Otáčky [ot/m]',
                title = 'Otáčky [ot/m]',
                figsize = (16, 8)
            )



# COMMAND ----------


pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df_1000['Rotational speed [rpm]'], 
                bins = 300, 
                x_name = 'Otáčky [ot/m]',
                title = 'Otáčky [ot/m]',
                figsize = (16, 8)
            )



# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Torque [Nm]'][50:], 
                bins = 50, 
                x_name = 'Kroutící moment [Nm]',
                title = 'Kroutící moment [Nm]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df_1000['Torque [Nm]'], 
                bins = 50, 
                x_name = 'Kroutící moment [Nm]',
                title = 'Kroutící moment [Nm]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Air temperature [K]'][50:], 
                bins = 50, 
                x_name = 'Teplota vzduchu [K]',
                title = 'Teplota vzduchu [K]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df_1000['Air temperature [K]'], 
                bins = 50, 
                x_name = 'Teplota vzduchu [K]',
                title = 'Teplota vzduchu [K]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Tool wear [min]'][50:], 
                bins = 50, 
                x_name = 'Délka opotřebení [min]',
                title = 'Délka opotřebení [min]',
                figsize = (16, 8)
            )



# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df_1000['Tool wear [min]'], 
                bins = 50, 
                x_name = 'Délka opotřebení [min]',
                title = 'Délka opotřebení [min]',
                figsize = (16, 8)
            )



# COMMAND ----------

df.columns

# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Process temperature [K]'][50:], 
                bins = 30, 
                x_name = 'Teplota procesu [K]]',
                title = 'Teplota procesu [K]',
                figsize = (16, 8)
            )



# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df_1000['Process temperature [K]'], 
                bins = 30, 
                x_name = 'Teplota procesu [K]]',
                title = 'Teplota procesu [K]',
                figsize = (16, 8)
            )

