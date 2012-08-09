import numpy as np 
from scipy.stats import scoreatpercentile
import pandas 

def summarize_continuous(d):
  stats = {}
  for xs in d.values():
    xs = np.array(xs)
    stats.setdefault('min', []).append(np.min(xs))
    stats.setdefault('max', []).append(np.max(xs))
    stats.setdefault('median', []).append(np.median(xs))
    stats.setdefault('lower_quartile', []).append(scoreatpercentile(xs, 25)) 
    stats.setdefault('upper_quartile', []).append(scoreatpercentile(xs, 75))
    stats.setdefault('count', []).append(len(xs))
  return pandas.DataFrame(stats, index = d.keys())

def summarize_bool(d):
  stats = {}
  for xs in d.values():
    xs = np.array(xs)
    stats.setdefault('mean', []).append(np.mean(xs))
    stats.setdefault('nz', []).append(np.sum(xs == 0))
    stats.setdefault('nnz', []).append(np.sum(xs == 1))
    stats.setdefault('count', []).append(len(xs))
  return pandas.DataFrame(stats, index = d.keys())
