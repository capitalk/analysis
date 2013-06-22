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

def outside_millisecond_range(hdf, start_time = None, end_time = None):
  return \
     (start_time and hdf.attrs['end_time'] < start_time) or \
     (end_time and hdf.attrs['start_time'] > end_time)
  
def compute_millisecond_indices(hdf, start_time = None, end_time = None):
  # extract the vector of millisecond timestamps, 
  # notice again that I slice into it to create a numpy array 
  t = hdf['t'][:]
  # do a binary search to find either the idx of start_time
  # or the first element greater than it
  if start_time: start_idx = np.searchsorted(t, end_time)
  else: start_idx = 0
  
  # do a binary search to find the idx past end_time 
  if end_time: end_idx = np.searchsorted(t, end_time, 'right') 
  else: end_idx = len(t)
  
  return start_idx, end_idx 


def hour_to_seconds(hour):
  if hour:
    return int(round(60 * 60 * hour))
  
def hour_to_milliseconds(hour):
  if hour:
    return int(round(hour * 1000 * 60 * 60  ))


def outside_hour_range(hdf, start_hour = None, end_hour = None):
  start_ms = hour_to_milliseconds(start_hour)
  end_ms = hour_to_milliseconds(end_hour)
  return outside_millisecond_range(hdf, start_ms, end_ms)
  
def compute_hour_indices(hdf, start_hour = None, end_hour = None):
  start_ms = hour_to_milliseconds(start_hour)
  end_ms = hour_to_milliseconds(end_hour)
  return compute_millisecond_indices(hdf, start_ms, end_ms)
  