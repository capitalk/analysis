import numpy as np
#import sklearn.ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
import math 
import cloud
import cloud_helpers 
import h5py 
from agg import mad, crossing_rate, rolling_fn 
import pandas
import scipy
import scipy.stats

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
  #'normalizer', 
  'past_lag',
  'transform'))
  
def copy_params(old_params, **kwds):
  param_args = {}
  for k in FeatureParams._fields:
    v = kwds[k] if k in kwds else getattr(old_params, k)
    param_args[k] = v
  return FeatureParams(**param_args)
  
def all_param_combinations(options, filter=lambda x: False):
  import itertools
  combinations = [x for x in apply(itertools.product, options.values())]
  params =[FeatureParams(**dict(zip(options.keys(), p))) for p in combinations]
  return [p for p in params if not filter(p)]
  
def gen_feature_params(raw_features=None):
  # raw_features is None means that each worker should figure out the common
  # feature set of its files 
  if raw_features is None:
    raw_features = [None]
  options = { 
    'raw_feature' : raw_features,
    'aggregator' : [np.median, np.std, crossing_rate],
     
     # window sizes in seconds
     'aggregator_window_size' : [10, 100, 1000], 
     #'normalizer': [None,  ZScore], # ZScore, LaplaceScore
     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None, 50, 200, 300, 400, 600, 1200, 6000], 
    
     'transform' : [None], 
  }
  def filter(x):
    return (x.past_lag and x.past_lag < x.aggregator_window_size)
  return all_param_combinations(options, filter = filter)
  
def construct_dataset(hdfs, features, future_offset, 
     start_hour = 3, end_hour = 7):
  inputs = []
  outputs = []
  # print "[construct_dataset] Future_offset = %s" % future_offset
  
  all_lags = [0] + [f.past_lag for f in features if f.past_lag is not None]
  all_aggregator_window_sizes = [0] + \
     [f.aggregator_window_size for f in features if f.aggregator_window_size is not None]
  
  max_lag = max(all_lags)
  max_aggregator_window_size = max(all_aggregator_window_sizes)

  for  hdf in hdfs:
    cols = []
    # construct all the columns for a subset of rows
    for param in features:
      """
      Get the raw feature from the HDF and then:
        (1) apply the rolling aggregator over the raw data
        (2) normalize the data
        (3) optionally transform the data
        (4) optionally get the percent change from some point in the past
      """
      x = hdf[param.raw_feature][:]
      n = len(x)
      x = rolling_fn(x, param.aggregator_window_size, param.aggregator)
      assert len(x) == n - param.aggregator_window_size  

      if param.transform: x = param.transform(x)
      lag = param.past_lag
      if lag:  
        past = x[:-lag]
        present = x[lag:]
        x = (present - past) 
      
      #print "Original column length for %s: %d" % (f, len(col))
      # remove last future_offset ticks, since we have no output for them
      x = x[:-future_offset]
      # skip some of the past if it's also seen by the feature with max. lag
      amt_lag_trim = max_lag - (0 if f.past_lag is None else f.past_lag)
      x = x[amt_lag_trim:]
      
      # if you're being aggregated in smaller windows than some other feature
      # then you should snip off some of your starting samples 
      amt_agg_window_trim = max_aggregator_window_size - \
        (0 if f.aggregator_window_size is None else f.aggregator_window_size)
      x = x[amt_agg_window_trim:]
      cols.append(x)
    mat = np.array(cols) 
    inputs.append(mat)
    # signal is: will the bid go up in some number of seconds
    bids = hdf['bid'][:]
    y = bids[future_offset:] > bids[:-future_offset]
    y = y[max_aggregator_window_size + max_lag:]
    outputs.append(y)
    
  inputs = np.hstack(inputs).T
  outputs = np.concatenate(outputs)
  return inputs, outputs
  
def common_features(hdfs):
  feature_set = None 
  for hdf in hdfs:
    curr_set = set(hdf.attrs['features'])
    if feature_set is None:
      feature_set = curr_set
    else:
      feature_set = feature_set.intersection(curr_set)
  for feature_name in feature_set:
    vec = hdf[feature_name]
    if np.all(vec == vec[0]):
      print "Skipping feature %s since it's constant %s" % (feature_name, vec[0])
      feature_set.remove(feature_name)
  return feature_set 

# affects only future self, no time travel but watch for self improvement
# and itching, but actually it just might be autoimmune or a mosquito 
# It's often hard to tell the difference. 
# To shed further light on this situation Timir should probably hire Sarah. 
# ...to do archival research for him. And a little bit of divination and/or
# creative writing. 
def eval_new_param(bucket, training_keys, testing_keys, old_params, new_param, 
    start_hour = 3, end_hour = 7, future_offset = 300):
  if new_param in old_params:
    return None

  n_train = len(training_keys)
  n_test = len(testing_keys)
  print "Downloading %d training HDFs..." % n_train
  training_hdfs = []
  def download(k):
    return cloud_helpers.download_file_from_s3(bucket, k) 
    
  for filename in cloud.mp.iresult(cloud.mp.map(download, training_keys)):
    print "Downloaded training file", filename
    hdf = h5py.File(filename)
    training_hdfs.append(hdf)
  
  print "Downloading %d testing HDFs..." % n_test
  testing_hdfs = [] 
  for filename in cloud.mp.iresult(cloud.mp.map(download, testing_keys)):
    print "Downloaded testing file", filename
    hdf = h5py.File(filename)
    testing_hdfs.append(hdf)
      
  if new_param.raw_feature is None:
    raw_features = common_features(training_hdfs)
  elif isinstance(new_param.raw_feature, list):
    raw_features = new_param.raw_feature
  else:
    raw_features = [new_param.raw_feature]
  print "Raw features: ", raw_features
  result = {}
  last_train = None
  for raw_feature in raw_features:
    param = copy_params(new_param, raw_feature = raw_feature)
    print param
    params = old_params + [param]
    x_train, y_train = \
      construct_dataset(training_hdfs, params, future_offset, start_hour, end_hour)
    assert last_train is None or np.any(last_train != x_train)
    last_train = x_train 
    
    x_test, y_test = \
      construct_dataset(testing_hdfs, params, future_offset, start_hour, end_hour)
    
    # print "x_train shape: %s, y_train shape: %s" % (x_train.shape, y_train.shape)
    # print "Training model..."
    if np.all(np.isfinite(x_train)):
      x_train_ok = True
    else:
      x_train_ok = False
      print "Training data contains NaN or infinity"
    if np.all(np.isfinite(x_test)):
      x_test_ok = True
    else:
      x_test_ok = False
      print "Testing data contains NaN or infinity"
    if np.all(np.isfinite(x_test)):
      y_test_ok = True
    else:
      y_test_ok = False
      print "Testing label contains NaN or infinity"
    if np.all(np.isfinite(x_test)):
      y_train_ok = True
    else:
      y_train_ok = False
      print "Testing label contains NaN or infinity"
    if x_train_ok and x_test_ok and y_train_ok and y_test_ok:
      model = LogisticRegression()
      model.fit(x_train, y_train)
      pred = model.predict(x_test)
      acc = np.mean(pred == y_test)
      print "Accuracy:", acc
    else:
      print "Skipping due to bad data", param 
      acc = None
    result[param] = acc
  print result 
  return result
    
def launch_jobs(hdf_bucket, training_keys, testing_keys, 
    raw_features = None, start_hour = 3, end_hour = 7, 
    run_local = False):
  all_params = gen_feature_params(raw_features)
  print "Launching %d jobs over %d training files and %d testing files" %  \
    (len(all_params), len(training_keys), len(testing_keys))
  old_chosen_params = []
  label = 'Evaluating %d parameter sets' % len(all_params)
  def do_work(p):
    return eval_new_param(hdf_bucket, training_keys, testing_keys,
      old_chosen_params, 
      p,  
      start_hour = start_hour, 
      end_hour = end_hour)
  mapper = cloud.mp.map if run_local else cloud.map 
  jids = mapper(do_work, 
    all_params, 
    _env = 'compute', 
    _label=label, 
    _type = 'f2')
  worst_acc = 1
  worst_param = None 
  best_acc = 0
  best_param = None
  results = {}
  for (i, result) in enumerate(cloud.iresult(jids)):
    print "Received result:" 
    # result can be 
    #  (1) None (if param was involid)
    #  (2) a single accuracy (if single parameter was sent)
    #  (3) a dictionary mapping parameters to accuracies 
    if result is None:
      result = {}
    else:
      assert isinstance(result, dict)
    for (param, acc) in result.items():
      print param, acc
      results[param]  = acc
      if acc and acc > best_acc:
        best_acc = acc
        best_param = param
      elif acc and acc < worst_acc:
        worst_acc = acc
        worst_param = param
      print "Current worst #%d: %s, acc = %s" % (i, worst_param, worst_acc)
      print "Current best #%d: %s, acc = %s" % (i, best_param, best_acc)
  return best_acc, best_param, worst_acc, worst_param, results

#def get_common_features(bucket, key_names):
#   features = None
#   for k in key_names: 
     
def single_feature_search(training_pattern, testing_pattern, 
    start_hour, end_hour, run_local):
  training_bucket, training_pattern = \
    cloud_helpers.parse_bucket_and_pattern(training_pattern)
  if len(training_pattern) == 0:
    training_pattern = '*'
  testing_bucket, testing_pattern = \
    cloud_helpers.parse_bucket_and_pattern(testing_pattern)
  if len(testing_pattern) == 0:
    testing_pattern = '*'
  assert training_bucket == testing_bucket 
  training_names = cloud_helpers.get_matching_key_names(training_bucket, training_pattern)
  testing_names = cloud_helpers.get_matching_key_names(testing_bucket, testing_pattern)
  return launch_jobs(training_bucket, training_names, testing_names, 
    raw_features = None, 
    start_hour = start_hour, 
    end_hour = end_hour, 
    run_local = run_local)
  

from argparse import ArgumentParser 
parser = ArgumentParser(description='Look for single best feature')
parser.add_argument('--train', type=str, dest='training_pattern', required=True,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--test', type=str, dest='testing_pattern', required=True, 
                        help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--run-local', dest="run_local", 
  action="store_true", default=False)
#parser.add_argument('--num-training-days', 
#  dest='num_training_days', type = int, default=16)
parser.add_argument('--start-hour', type = int, default = 3, dest="start_hour")
parser.add_argument('--end-hour', type = int, default = 7, dest="end_hour")
#parser.add_argument('--min-duration', dest='min_dur', type=int, default=None, 
#  help  = 'ignore files which ')

if __name__ == '__main__':
  args = parser.parse_args()
  #assert args.pattern 
  #assert len(args.pattern) > 0
  best_acc, best_param, worst_acc, worst_param, results = \
    single_feature_search(args.training_pattern, args.testing_pattern,  
     args.start_hour, args.end_hour, args.run_local)
  print results
  print "Worst param: %s, accuracy = %s" % (worst_param, worst_acc)
  print "Best param: %s, accuracy = %s" % (best_param, best_acc)
  