import numpy as np 
from scipy.stats import scoreatpercentile
import pandas 

def summarize_dict(d):
  stats = {}
  for xs in d.values():
    stats.setdefault('min', []).append(np.min(xs))
    stats.setdefault('max', []).append(np.max(xs))
    stats.setdefault('median', []).append(np.median(xs))
    stats.setdefault('lower_quartile', []).append(scoreatpercentile(xs, 25)) 
    stats.setdefault('upper_quartile', []).append(scoreatpercentile(xs, 75))
  return pandas.DataFrame(stats, index = d.keys())
