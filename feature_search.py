import numpy as np
import sklearn
import sklearn.ensembles
from sklearn.ensembles import RandomForestClassifier
import math 
import cloud


def mad(x):
  """median absolute deviation from the sample median"""
  return np.median(np.abs(x - np.median(x)))

def crossing_rate(x):
  """# of times the series crosses its own initial value / length of series"""
  init = x[0]
  gt = x > init
  lt = x <= init
  moves_down = gt[:-1] & lt[1:]
  moves_up = lt[:-1] & gt[1:]
  n_crosses = np.sum(moves_down) + np.sum(moves_up)
  return n_crosses / float(len(x))
  
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
    self.mad = np.median(np.abs(x - m))
  
  def transform(self, x):
    return (x - self.median) / self.mad

class RescaleExtrema:
  def __init__(self):
    self.min = None
    self.max = None
  
  def fit(self, x):
    self.min = np.min(x)
    self.range = np.max(x) - self.min
    
  def transform(self, x):
    return (x - self.min) / self.range

def random_subset_with_sequential_dependency(x, p):
  """
  We want to choose n*p of the total n elements in x, but
  we want the probability of x[i] in S to be high if x[i-1]
  is in x and low if x[i-1] is not in x. One way to do this is:
    p(include x_i) = p^2 if x_(i-1) not included 
                   -or- sqrt(p) if x_(i-1) was included
    can we prove that the expectation is still p*n?    
    
  FROM EXPERIMENTS I SEE THAT THIS DOESN'T WORK--- WE GET TOO FEW
  RESULTS BACK, SHOULD TRY SOMETHING ELSE!
  """
    
  subset = []
  # initial state is randomly set to include or exclude 
  rand = np.random.rand
  high_p = np.sqrt(p)
  low_p = p**1.5
  state = rand() < p
  for (i, xi) in enumerate(x):
    if state: 
      subset.append(xi)
      state = rand() < high_p
    else:
      state = rand() < low_p
  return subset 

def sqrt(x):
  return x * x

"""
A feature consists of: 
- a raw feature 
- an optional normalization scheme, such as z-score or (x-min) / (max-min)
- an aggregator function (i.e. median, mean, var) 
- the aggregator's window size (how many ticks back do you look)
- optionally, a past lag relative to which we measure percent change 
"""


from collections import namedtuple
FeatureParams = namedtuple('FeatureParams', 
  ('raw_feature', 
  'aggregator', 
  'aggregator_window_size', 
  'normalizer', 
  'past_lag',
  'transform'))
  
def all_param_combinations(options, filter=lambda x: False):
  import itertools
  combinations = [x for x in apply(itertools.product, options.values())]
  params =[FeatureParams(**dict(zip(options.keys(), p))) for p in combinations]
  return [p for p in params if not filter(p)]
  
  
def gen_feature_params(raw_features=['bid', 'offer']):
  options = { 
    'raw_feature' : raw_features,
    
    'aggregator' : [np.median, mad, crossing_rate],
     
     # window sizes in seconds
     'aggregator_window_size' : [1, 10, 100], 
     'normalizer': [None, ZScore,  RescaleExtrema], # LaplaceScore,

     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None, 5, 20, 30, 40, 60, 120, 600],
    
     'transform' : [None, np.square]
  }
  return all_param_combinations(options, filter = lambda x: x.past_lag and x.past_lag < x.aggregator_window_size)

def extract_feature(hdf, param, normalizer = None):
  x = hdf[param.raw_feature][:]
  n = len(x)
  
  if normalizer:
    N = normalizer
  elif param.normalizer:
    N = param.normalizer()
    N.fit(x)
  else:
    N = None 
    
  if N:
    x = N.transform(x)
    
  # window size is in seconds, but each tick is 100ms
  w = 10 * param.aggregator_window_size
  n_agg = n - w
  aggregated = np.zeros(n_agg)
  agg_fn = param.aggregator
  for i in xrange(len(aggregated)):
    aggregated[i] = agg_fn(x[i:i+w])
  
  lag = params.past_lag
  if not lag:
    lagged = aggregated
  
  else:
    # like window size, lag is in seconds, but ticks are 100ms
    future = aggregated[10*lag:]
    present  = aggregated[:-10*lag])
    diff = future - present
    lagged = 100 * diff / present 
    
  if param.transform:
    result = param.transform(lagged)
  else:
    result = lagged
  return result, N

def fit_normalizers(hdfs, features):
  normalizers = []
  num_files = len(hdfs)
  half = int(math.ceil(num_files / 2.0))
  
  for (i, f) in features:
    if f.normalizer is None:
      normalizers.append(None)
    else:
        # only use first half of dataset for estimating normalizer params
        # so we don't overfit by normalization
      data = []
      for hdf in hdfs[:half]:
        col = hdf[f.raw_feature][:]
        data.append(col)
      data = np.concatenate(data)
      N = f.normalizer()
      N.fit(data)
      normalizers.append(N)
  return normalizers 
  
def construct_dataset(hdfs, features, future_offset, normalizers = None):
  inputs = []
  outputs = []
  # future offset is in seconds, ticks are 100ms
  future_offset_ticks = 10 * future_offset
  if normalizers is None:
    normalizers = fit_normalizers(hdfs, features)
  
  for hdf in hdfs:
    bids = hdf['bid']
    # signal is: will the bid go up in some number of seconds
    y = bids[future_offset_ticks:] > bids[:-future_offset_ticks]
    outputs.append(y)
     
    cols = []
    # construct all the columns for a subset of rows
    for (i, f) in enumerate(features):
      N = normalizers[i]
      col = apply_feature_params(hdf, f, N)
       # remove last future_offset*10 ticks, since we have no output for them
      col = col[:-future_offset_ticks]
      cols.append(col)
    mat = np.hstack(cols)
    inputs.append(mat)
  return np.vstack(inputs), np.concatenate(outputs), normalizer
  
def download_hdfs(bucket, keys):
  hdfs = []
  for key in keys:
    print "Downloading", key
    hdf = cloud_helpers.download_hdf_from_s3(bucket, key)
  return hdfs 
  
import sklearn
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier 

def eval_new_param(hdf_bucket, hdf_keys, old_params, new_param, 
    n_training = 14, future_offet = 30):
  params = old_params + [new_param]
  n_files = len(hdf_keys)
  
  accs = []
  for test_idx = np.arange(n_files)[n_training:]
    training_filenames = hdf_keys[(test_idx - n_training):test_idx]
    test_filename = test_idx
    training_hdfs = download_hdfs(hdf_bucket, training_filenames)
    test_hdf = cloud_helpers.download_hdf_from_s3(bucket, test_filename)
    x_train, y_train, normalizer = construct_dataset(training_hdfs, params, future_offset)
    x_test, y_test, _ = construct_dataset([test_hdf], params, normalizer)
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    acc = np.mean(pred == y_test)
    accs.append(acc)
  return np.median(acc)
    
def launch_jobs(hdf_bucket, hdf_keys, raw_features):
  all_params = gen_feature_params(raw_features)
  print "Launching %d jobs" % len(all_params)
  jids = cloud.map(lambda p: eval_extra_param(hdf_bucket, hdf_keys, [], p), all_params)
  worst_acc = 1
  worst_param = None 
  best_acc = 0
  best_param = None
  for (i, acc) in enumerate(cloud.iresult(jids)):
    if acc > best_acc:
      best_acc = acc
      best_param = all_params[i]
    elif acc < worst_acc:
      worst_acc = acc
      worst_param = all_params[i]
  return best_acc, best_param, worst_acc, worst_param
