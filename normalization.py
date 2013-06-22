import numpy as np 

class ZScore:
  def __init__(self):
    self.mean = None 
    self.std = None
  
  def fit(self, data ):
    self.mean = np.mean(data)
    self.std = np.std(data)
  
  def transform(self, x):
    return (x - self.mean) / self.std
    
class LaplaceScore:
  def __init__(self):
    self.median = None
    self.mad = None
    
  def fit(self, x):
    m = np.median(x)
    self.median = m
    self.mad = np.mean(np.abs(x - m))
  
  def transform(self, x):
    return (x - self.median) / self.mad

class RescaleExtrema:
  def __init__(self):
    self.min = None
    self.max = None
  
  def fit(self, x):
    low = np.min(x)
    high = np.max(x)
    self.min = low
    self.range = high - low 
    
  def transform(self, x):
    return (x - self.min) / self.range