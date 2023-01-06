# Databricks notebook source
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