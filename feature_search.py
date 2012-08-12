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
    low = scipy.stats.scoreatpercentile(x, 10)
    high = scipy.stats.scoreatpercentile(x, 90)
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
  'normalizer', 
  'past_lag',
  'transform'))
  
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
    
    'aggregator' : [np.median, mad, crossing_rate], #[np.median]
     
     # window sizes in seconds
     'aggregator_window_size' : [10, 100, 1000], # [10],
     'normalizer': [None, ZScore, LaplaceScore], #[LaplaceScore]

     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None, 50, 200, 300, 400, 600, 1200, 6000], #[None], #[
    
     'transform' : [None, np.square], # [np.square], #
  }
  def filter(x):
    return \
      (x.past_lag and x.past_lag < x.aggregator_window_size) or \
      (x.normalizer is not None and x.past_lag is not None)
  return all_param_combinations(options, filter = filter)



def extract_feature(hdf, param,  normalizer = None):
  """
  Get the raw feature from the HDF and then:
    (1) apply the rolling aggregator over the raw data
    (2) normalize the data
    (3) optionally transform the data
    (4) optionally get the percent change from some point in the past
  """
  x = hdf[param.raw_feature][:]
  n = len(x)
  #print "original", np.sum(x== 0), len(x) - np.sum(np.isfinite(x)), x[-20:] 
  x = rolling_fn(x, param.aggregator_window_size, param.aggregator)
  #print "agg", np.sum(x== 0), len(x) - np.sum(np.isfinite(x)), x[-20:]
  
  assert len(x) == n - param.aggregator_window_size  
  
  if normalizer:
    N = normalizer
  elif param.normalizer:
    N = param.normalizer()
    N.fit(x)
  else:
    N = None     
  
  if N: x = N.transform(x)
  #print "normalized", np.sum(x== 0), len(x) - np.sum(np.isfinite(x)), x[-20:]
  
  if param.transform: x = param.transform(x)
  #print "transformed", np.sum(x== 0), len(x) - np.sum(np.isfinite(x)), x[-20:]
  
  lag = param.past_lag
  if lag:  
    past = x[:-lag]
    present = x[lag:]
    x = (present - past) #/ past
  #print "lag", np.sum(x== 0),  len(x) - np.sum(np.isfinite(x)), x[-20:]
  return x, N

def fit_normalizers(hdfs, features):
  normalizers = []
  num_files = len(hdfs)
  half = int(math.ceil(num_files / 2.0))
  
  for (i, f) in enumerate(features):
    if f.normalizer is None:
      normalizers.append(None)
    else:
        # only use first half of dataset for estimating normalizer params
        # so we don't overfit by normalization
      data = []
      for hdf in hdfs[:half]:
        col = hdf[f.raw_feature][:]
        if param.aggregator:
          col = rolling_fn(col, param.aggregator_window_size, param.aggregator)
        data.append(col)
      data = np.concatenate(data)
      N = f.normalizer()
      N.fit(data)
      normalizers.append(N)
  return normalizers 
  
def construct_dataset(hdfs, features, future_offset, start_hour, end_hour,
     normalizers = None, cached_inputs = {}, cached_outputs = {}):
  inputs = []
  outputs = []
  print "[construct_dataset] Future_offset = %s" % future_offset
  
  all_lags = [0] + [f.past_lag for f in features if f.past_lag is not None]
  all_aggregator_window_sizes = [0] + \
     [f.aggregator_window_size for f in features if f.aggregator_window_size is not None]
  
  max_lag = max(all_lags)
  max_aggregator_window_size = max(all_aggregator_window_sizes)
  
  # TODO: Actually use start_hour and end_hour 
  if normalizers is None:
    normalizers = fit_normalizers(hdfs, features)
  
  for (i, hdf) in enumerate(hdfs):
    if hdf.filename in cached_inputs:
      mat = cached_inputs[hdf.filename]
    else: 
      cols = []
      # construct all the columns for a subset of rows
      for (i, f) in enumerate(features):
        N = normalizers[i]
        col, _ = extract_feature(hdf, f, N)
        #print "Original column length for %s: %d" % (f, len(col))
        # remove last future_offset ticks, since we have no output for them
        col = col[:-future_offset]
        # skip some of the past if it's also seen by the feature with max. lag
        amt_lag_trim = max_lag - (0 if f.past_lag is None else f.past_lag)
        col = col[amt_lag_trim:]
        
        # if you're being aggregated in smaller windows than some other feature
        # then you should snip off some of your starting samples 
        amt_agg_window_trim = max_aggregator_window_size - \
          (0 if f.aggregator_window_size is None else f.aggregator_window_size)
        col = col[amt_agg_window_trim:]
        #print "Final column length for %s : %d" % (f, len(col))
        cols.append(col)
      mat = np.array(cols)
      cached_inputs[hdf.filename] = mat
    
    inputs.append(mat)
    # signal is: will the bid go up in some number of seconds
    if hdf.filename in cached_outputs:
      y = cached_outputs[hdf.filename]
    else:
      bids = hdf['bid'][:]
      y = bids[future_offset:] > bids[:-future_offset]
      y = y[max_aggregator_window_size + max_lag:]
      cached_outputs[hdf.filename] = y
    outputs.append(y)
    #print "%d, feature shape: %s, output shape: %s" % (i, mat.shape, y.shape)
  
  inputs = np.hstack(inputs).T
  outputs = np.concatenate(outputs)
  return inputs, outputs, normalizers 
  
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
def eval_new_param(bucket, hdf_keys, old_params, new_param, 
    num_training_days = 16, start_hour = 3, end_hour = 7, future_offset = 300):
  if new_param in old_params:
    return None

  n_files = len(hdf_keys)
  print "Downloading all HDFs..."
  
  hdfs = []
  if True:
    jids = cloud.mp.map(lambda k: cloud_helpers.download_file_from_s3(bucket, k), hdf_keys)
    for (i, filename) in enumerate(cloud.mp.iresult(jids, num_in_parallel = 10)):
      print "Done downloading file #%d: %s" % (i, filename)
      hdf = h5py.File(filename)
      hdfs.append(hdf)
  else:
    for k in hdf_keys:
      print "Downloading %s:%s" % (bucket, k)
      hdf = cloud_helpers.download_hdf_from_s3(bucket, k)
      hdfs.append(hdf)
      
  if new_param.raw_feature is None:
    raw_features = common_features(hdfs)
  else:
    raw_features = [new_param.feature]
  print "Raw features: ", raw_features
  result = {}
  for raw_feature in raw_features:
    param = FeatureParams(raw_feature = raw_feature, 
      aggregator = new_param.aggregator, 
      aggregator_window_size = new_param.aggregator_window_size, 
      normalizer = new_param.normalizer, 
      past_lag = new_param.past_lag, 
      transform = new_param.transform)
    print param
    params = old_params + [param]
    accs = []
    for test_idx in np.arange(n_files)[num_training_days:]:
      training_hdfs = hdfs[(test_idx - num_training_days):test_idx]
      test_hdf = hdfs[test_idx]
      print "Constructing dataset for %d / %d (test_filename = %s)" % \
        (test_idx, n_files, test_hdf.filename)
      x_train, y_train, normalizers = \
        construct_dataset(training_hdfs, params, future_offset, start_hour, end_hour, None)
      x_test, y_test, _ = \
        construct_dataset([test_hdf], params, future_offset, start_hour, end_hour, normalizers)
      print "x_train shape: %s, y_train shape: %s" % (x_train.shape, y_train.shape)
      # print "Training model..."
      if np.all(np.isfinite(x_train)) and np.all(np.isfinite(x_test)) and \
          np.all(np.isfinite(y_train)) and np.all(np.isfinite(y_test)):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        # print "Generating predictions"
        pred = model.predict(x_test)
        acc = np.mean(pred == y_test)
        print "Accuracy for %d / %d = %s" % (test_idx, n_files, acc)
        accs.append(acc)
      else:
        print "Skipping test #%s of %s due to NaN or infinity in data" %  (test_idx, param)
    if len(accs) > 0:
      med_acc = np.median(accs)
    else:
      med_acc = None
    print "Median accuracy: %s" % med_acc
    result[param] = med_acc
  print result 
  return result
    
def launch_jobs(hdf_bucket, hdf_keys, raw_features = None, 
    num_training_days = 16, start_hour = 3, end_hour = 7, run_local = False):
  #if raw_features is None:
  #  print "Downloading ", hdf_keys[0]
  #  hdf = cloud_helpers.download_hdf_from_s3(hdf_bucket, hdf_keys[0])
  #  raw_features = hdf.attrs['features']
  #print "Raw features", raw_features
  
  all_params = gen_feature_params(raw_features)
  print "Launching %d jobs" % len(all_params)
  old_chosen_params = []
  label = 'Evaluating %d parameter sets' % len(all_params)
  def do_work(p):
    return eval_new_param(hdf_bucket, hdf_keys, old_chosen_params, p, 
      num_training_days = num_training_days, 
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
    print "Received result #%d: %s" % (i, result)
    # result can be 
    #  (1) None (if param was involid)
    #  (2) a single accuracy (if single parameter was sent)
    #  (3) a dictionary mapping parameters to accuracies 
    if result is None:
      result = {}
    elif not isinstance(result, dict):
      result = {all_params[i]: result}
    for (param, acc) in result.items():
      results[param]  = acc
      if acc and acc > best_acc:
        best_acc = acc
        best_param = all_params[i]
      elif acc and acc < worst_acc:
        worst_acc = acc
        worst_param = all_params[i]
  return best_acc, best_param, worst_acc, worst_param, results

#def get_common_features(bucket, key_names):
#   features = None
#   for k in key_names: 
     
def single_feature_search(pattern, num_training_days, start_hour, end_hour, run_local):
  bucket, key_pattern = cloud_helpers.parse_bucket_and_pattern(pattern)
  if len(key_pattern) == 0:
    key_pattern = '*'
  key_names = cloud_helpers.get_matching_key_names(bucket, key_pattern)
  #raw_features = get_common_features(bucket, key_names)
  #print "Raw features:", raw_features
  return launch_jobs(bucket, key_names, raw_features = None, 
    num_training_days = num_training_days, 
    start_hour = start_hour, 
    end_hour = end_hour, 
    run_local = run_local)
  

from argparse import ArgumentParser 
parser = ArgumentParser(description='Look for single best feature')
parser.add_argument('pattern', metavar='P', type=str,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--run-local', dest="run_local", 
  action="store_true", default=False)
parser.add_argument('--num-training-days', 
  dest='num_training_days', type = int, default=16)
parser.add_argument('--start-hour', type = int, default = 3, dest="start_hour")
parser.add_argument('--end-hour', type = int, default = 7, dest="end_hour")
#parser.add_argument('--min-duration', dest='min_dur', type=int, default=None, 
#  help  = 'ignore files which ')

if __name__ == '__main__':
  args = parser.parse_args()
  assert args.pattern 
  assert len(args.pattern) > 0
  best_acc, best_param, worst_acc, worst_param, results = \
    single_feature_search(args.pattern, args.num_training_days, 
    args.start_hour, args.end_hour, args.run_local)
  print results
  print "Worst param: %s, accuracy = %s" % (worst_param, worst_acc)
  print "Best param: %s, accuracy = %s" % (best_param, best_acc)
  