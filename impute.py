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
  
    if (tol and name) is not None:
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
    
    #self.save_figure(
    #              distribution = distribution,
    #              tol = tol,
    #              name = name
    #            )

  
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


class ImpError:
  def __init__(self, actual: np.array, imputed: np.array):
    self.actual = actual
    self.imputed = imputed
    
   
  def _calculate_error_point(self, name, error_lst = []):
    for actual, imputed in list(zip(self.actual, self.imputed)):
      err = np.abs((imputed - actual)/actual)
      error_lst.append(err)

      #print(f"Actual: {actual} \t Imputed: {imputed} \t err: {err} \n")
      #np.savetxt(f"/dbfs/mnt/pbi/Bots/err_{name}.csv", error_lst, delimiter=",")
      
      
      
    return sum(error_lst) / len(self.imputed)
  



EM_SKUTECNA = "Skutečná hodnota"
EM_ODHADNUTA = "Hodnota odhanutá EM algoritmem"
CHYBA_ODHADU = "Chyba odhadu chybějící hodnoty"


col = 'Teplota vzduchu'

df = pd.read_csv(f"./data/x_train_miss.csv")



copy_arr_air_temp = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_normal_air_temp, error_lst_normal_air_temp, index_lst_normal_air_temp = impute.em_imputation(distribution = "normal", tol = 0.001)

copy_arr_air_temp = copy_arr_air_temp.to_numpy()

error_normal = ImpError(
  actual = copy_arr_air_temp[0:50],
  imputed = pred_normal_air_temp[0:50]
)

error_normal._calculate_error_point(name = col)

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )


pplot.imp_plot(
               y = copy_arr_air_temp[0:50],
               y2 = pred_normal_air_temp[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA
              )

df[col][0:50] = pred_normal_air_temp[0:50]
df.to_csv('x_train_miss.csv')

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

col = 'Kroutící moment'

df = pd.read_csv(f"./data/x_train_miss.csv")


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

df[col][0:50] = pred_normal_torque[0:50]
df.to_csv('x_train_miss.csv')

error_normal._calculate_error_point(name = col)

# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
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


col = 'Rychlost otáček'

df = pd.read_csv(f"./data/x_train_miss.csv")

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

print(error_normal._calculate_error_point(name = col))

distr = DistributionInfo(vals = df[col])
distr.get_kurtosis_skewness()  


pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
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

from sklearn.model_selection import train_test_split

col = 'Rychlost otáček'

df = pd.read_csv(f"./data/x_train.csv")

#################################################################################################################################
from sklearn.model_selection import train_test_split
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


print(error_normal_tol._calculate_error_point(name = col))


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

df[col][0:50] = pred_skewed_normal[0:50]
df.to_csv('x_train_miss.csv')



col = 'Opotřebení nástroje'


df = pd.read_csv(f"./data/x_train.csv")



copy_arr_tool_wear = df[col][0:50].copy()
df[col][0:50] = np.nan

impute = EM_impute(df[col].to_numpy())
pred_normal, error_lst, index_lst_normal = impute.em_imputation(distribution = "normal", tol = 0.001)

copy_arr_tool_wear = copy_arr_tool_wear.to_numpy()

error_uniform = ImpError(
  actual = copy_arr_tool_wear[0:50],
  imputed = pred_normal[0:50]
)

df[col][0:50] = pred_normal[0:50]
df.to_csv('x_train_miss.csv')

error_uniform._calculate_error_point(name = col)

# COMMAND ----------

pplot = ImpPlot(
                SMALL_SIZE = 20, 
                BIG_SIZE = 16
            )

pplot.imp_plot(
               y = copy_arr_tool_wear[0:50],
               y2 = pred_normal[0:50],
               y_color = 'o:r',
               y_label = EM_SKUTECNA,
               y2_color = 'o:b',
               y2_label = EM_ODHADNUTA, 
               distribution = "normal", 
               tol = '0.01', 
               units = "[ot/min]",
               name = "opotrebeni3"
              )

import numpy as np
import pandas as pd
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv(f"./data/x_train_miss.csv")

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('fivethirtyeight')
test = df['Rychlost otáček']

SAMPLE_SIZE = 10000
N_BINS = 300


pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Rychlost otáček'][50:], 
                bins = 300,
                x_name = 'Otáčky [ot/m]',
                title = 'Otáčky [ot/m]',
                figsize = (16, 8)
            )



pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Kroutící moment'][50:], 
                bins = 50, 
                x_name = 'Kroutící moment [Nm]',
                title = 'Kroutící moment [Nm]',
                figsize = (16, 8)
            )



pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Teplota vzduchu'][50:], 
                bins = 50, 
                x_name = 'Teplota vzduchu [K]',
                title = 'Teplota vzduchu [K]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Teplota vzduchu'][50:], 
                bins = 50, 
                x_name = 'Teplota vzduchu [K]',
                title = 'Teplota vzduchu [K]',
                figsize = (16, 8)
            )


# COMMAND ----------

pplot = ImpPlot(SMALL_SIZE = 20, 
                BIG_SIZE = 16)
pplot.histplot(
                data = df['Opotřebení nástroje'][50:], 
                bins = 50, 
                x_name = 'Délka opotřebení [min]',
                title = 'Délka opotřebení [min]',
                figsize = (16, 8)
            )

