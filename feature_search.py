import numpy as np
#import sklearn.ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
import math 
import cloud
import cloud_helpers 
import h5py 
from agg import mad, crossing_rate, rolling_fn 
from normalization import ZScore 
import pylab

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

def xor_none(x,y):
  return (x is None and y is not None) or (x is not None and y is None)
  
def gen_feature_params(raw_features=None):
  # raw_features is None means that each worker should figure out the common
  # feature set of its files 
  if raw_features is None:
    raw_features = [None]
  options = { 
    'raw_feature' : raw_features,
    'aggregator' : [None, np.mean, np.std], #, crossing_rate],
     
     # window sizes in seconds
     'aggregator_window_size' : [None, 100], 
     'normalizer': [ZScore], # ZScore, LaplaceScore
     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None, 50, 200, 300, 400, 600, 1200, 6000], 
    
     'transform' : [None], 
  }
  def filter(x):
    return (x.past_lag and x.past_lag < x.aggregator_window_size) or \
      xor_none(x.aggregator, x.aggregator_window_size)
  return all_param_combinations(options, filter = filter)
  
def construct_dataset(hdfs, features, future_offset, 
     start_hour = 3, end_hour = 7):
  inputs = []

  all_lags =[f.past_lag for f in features if f.past_lag is not None]
  max_lag = max([0] + all_lags)
  all_aggregator_window_sizes = \
     [f.aggregator_window_size for f in features if f.aggregator_window_size is not None]
  
  max_aggregator_window_size = max([0] + all_aggregator_window_sizes)
  print "max_lag", max_lag
  print "max_agg_window", max_aggregator_window_size
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
      raw_feature = param.raw_feature
      x = hdf[raw_feature][:]
      assert np.all(np.isfinite(x))
      if 'vol' in raw_feature:
        x /= 10.0 ** 6
      elif raw_feature == 't':
        x /= float(x[-1])
      assert np.all(np.isfinite(x)), "Raw features %s contains bad data" % raw_feature
      #n = len(x)
      w = param.aggregator_window_size
      x = rolling_fn(x, w, param.aggregator)[w:]
      assert np.all(np.isfinite(x)), \
        "Got bad data from rolling aggregator %s (win size = %d)" %\
           (param.aggregator, param.aggregator_window_size) 

      if param.transform: x = param.transform(x)
      assert np.all(np.isfinite(x))
      lag = param.past_lag
      if lag:  
        past = x[:-lag]
        present = x[lag:]
        x = (present - past) 
    
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
    assert np.all(np.isfinite(mat))
    inputs.append(mat)
  total_lag = max_aggregator_window_size + max_lag
  X = np.hstack(inputs).T
  return X, total_lag 

def construct_outputs(hdfs, future_offset, lag = 0):
  outputs = []
  for hdf in hdfs:
    # signal is: will the bid go up in some number of seconds
    bids = hdf['bid'][:]
    offers = hdf['offer'][:]
    midprice = (bids+offers)/2.0
    y = np.sign(midprice[future_offset:] - midprice[:-future_offset])
    y = y[lag:]
    assert np.all(np.isfinite(y))
    outputs.append(y)
  return np.concatenate(outputs)
  
def normalize_data(x, params = None, normalizers = None):
  assert params or normalizers
  cols = np.zeros_like(x)
  if params is not None:
    normalizers = []
    for (i, p) in enumerate(params):
      col = x[:, i]
      n = p.normalizer
      if n:
        n = n()
        n.fit(x)
        col = n.transform(col)
      cols[:, i] = col 
      normalizers.append(n)
    return cols, normalizers
  else:
    for (i, n) in enumerate(normalizers):
      col = x[:, i]
      if n: col = n.transform(col)
      cols[:, i] = col 
    return cols
     
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

def eval_params(training_hdfs, testing_hdfs, old_params, new_param, start_hour, end_hour, future_offset):
  if new_param.raw_feature is None:
    raw_features = common_features(training_hdfs)
  elif isinstance(new_param.raw_feature, list):
    raw_features = new_param.raw_feature
  else:
    raw_features = [new_param.raw_feature]
  print "Raw features: ", raw_features
  result = {}
  last_train = None
  y_train = None
  y_test = None 
  for (i, raw_feature) in enumerate(raw_features):
    param = copy_params(new_param, raw_feature = raw_feature)

    print param
    if param in old_params:
      print "...duplicate param in dataset, skipping..."
      result[param] = None
    elif param.raw_feature == 't' and param.aggregator not in [np.mean, np.median, None]:
      print "raw_feature = time with aggregator = %s, skipping..." % param.aggregator
      result[param] = None
    else:  
      params = old_params + [param]
      x_train, lag = \
        construct_dataset(training_hdfs, params, future_offset, start_hour, end_hour)
      if y_train is None:
        y_train = construct_outputs(training_hdfs, future_offset, lag)
        y_test = construct_outputs(testing_hdfs, future_offset, lag)

      if last_train is not None and np.all(last_train == x_train):
        "WARNING: Got identical data as last iteration for " + raw_feature
      x_train, normalizers = normalize_data(x_train, params = params)
      print "x_train shape: %s, y_train shape: %s" % (x_train.shape, y_train.shape)
      assert normalizers is not None
      last_train = x_train 
      x_test, _ = \
       construct_dataset(testing_hdfs, params, future_offset, start_hour, end_hour)
      x_test = normalize_data(x_test, normalizers = normalizers)
      print "x_test shape: %s, y_test shape: %s" % (x_test.shape, y_test.shape)

      # print "Training model..."
      if np.all(np.isfinite(x_train)): x_train_ok = True
      else:
        x_train_ok = False
        print "Training data contains NaN or infinity"
      if np.all(np.isfinite(x_test)): x_test_ok = True
      else:
        x_test_ok = False
        print "Testing data contains NaN or infinity"
      if np.all(np.isfinite(y_train)): y_train_ok = True
      else:
        y_train_ok = False
        print "Training label contains NaN or infinity"
        
      if np.all(np.isfinite(y_test)): y_test_ok = True
      else:
        y_test_ok = False
        print "Testing label contains NaN or infinity"
      if x_train_ok and x_test_ok and y_train_ok and y_test_ok:
        #model = LogisticRegression()
        model = DecisionTreeClassifier(max_depth = 3)  
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        n_neg = np.sum(pred <= 0)
        n_pos = np.sum(pred > 0)
        acc = np.mean(pred == y_test)
        print "neg = %d, pos = %d, pred neg = %d, pred pos = %d, accuracy = %s" % \
          (np.sum(y_test <= 0), np.sum(y_test > 0), n_neg, n_pos, acc)
        result[param] = acc
        print 
      else:
        print "Skipping due to bad data", param 
        result[param] = None
        print 
  return result

# affects only future self, no time travel but watch for self improvement
# and itching, but actually it just might be autoimmune or a mosquito 
# It's often hard to tell the difference. 
# To shed further light on this situation Timir should probably hire Sarah. 
# ...to do archival research for him. And a little bit of divination and/or
# creative writing. 
def download_and_eval(bucket, training_keys, testing_keys, old_params, new_param, 
    start_hour = 3, end_hour = 7, future_offset = 300):
 
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
  result = \
    eval_params(training_hdfs, testing_hdfs, old_params, new_param, start_hour, end_hour, future_offset)    
  print result
  return result
    
def launch_jobs(hdf_bucket, training_keys, testing_keys, 
    raw_features = None, start_hour = 3, end_hour = 7, 
    num_features = 1):
  all_possible_params = gen_feature_params(raw_features)

  label = 'Evaluating %d parameter sets' % len(all_possible_params)
  chosen_params = []
  def worker_wrapper(new_param):
    return download_and_eval(hdf_bucket, training_keys, testing_keys,
      chosen_params, 
      new_param,  
      start_hour = start_hour, 
      end_hour = end_hour)
  
  for feature_num in xrange(num_features):
    worst_acc = 1
    worst_param = None 
    best_acc = 0
    best_param = None
    print "=== Searching for feature #%d ===" % feature_num
    print "Launching %d jobs over %d training files and %d testing files" %  \
      (len(all_possible_params), len(training_keys), len(testing_keys))

    jids =\
      cloud.map(worker_wrapper, all_possible_params, 
        _env = 'compute', 
        _label=label, 
        _type = 'f2')
    results = {}
    for (i, result) in enumerate(cloud.iresult(jids)):
      print "Received result:" 
      if result is None:
        result = {}
      else:
        assert isinstance(result, dict)
      for (param, acc) in result.items():
        print param, acc
        results[tuple(chosen_params + [param])]  = acc
        if acc and acc > best_acc:
          best_acc = acc
          best_param = param
        elif acc and acc < worst_acc:
          worst_acc = acc
          worst_param = param
    print "Current worst after result #%d for feature #%d: %s, acc = %s" % \
      (i, feature_num, worst_param, worst_acc)
    print "Current best after result #%d for feature #%d: %s, acc = %s" % \
      (i, feature_num, best_param, best_acc)
    chosen_params.append(best_param)
  return chosen_params, best_acc, worst_acc, worst_param, results

     
def collect_keys_and_launch(training_pattern, testing_pattern, 
    start_hour = 3, end_hour = 7, num_features = 1):
  # if files are local then buckets are None
  # otherwise we expect HDFs to be on the same S3 bucket 
  
  if training_pattern.startswith('s3'):
    training_bucket, training_pattern = \
      cloud_helpers.parse_bucket_and_pattern(training_pattern)
    if len(training_pattern) == 0:
      training_pattern = '*'
    training_names = cloud_helpers.get_matching_key_names(training_bucket, training_pattern)
  else:
    assert False
    #print "Local training files: ", training_pattern
    #training_bucket = None 
  
  if testing_pattern.startswith('s3'):
    testing_bucket, testing_pattern = \
      cloud_helpers.parse_bucket_and_pattern(testing_pattern)
    if len(testing_pattern) == 0:
      testing_pattern = '*'
    testing_names = cloud_helpers.get_matching_key_names(testing_bucket, testing_pattern)
  else:
    assert False   
    
  assert training_bucket == testing_bucket, \
    "Expected training bucket %s to be same as testing bucket %s" % (training_bucket, testing_bucket)
 
  return launch_jobs(training_bucket, training_names, testing_names, 
    raw_features = None, 
    start_hour = start_hour, 
    end_hour = end_hour, 
    num_features = num_features)
  

from argparse import ArgumentParser 
parser = ArgumentParser(description='Look for single best feature')
parser.add_argument('--train', type=str, dest='training_pattern', required=True,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--test', type=str, dest='testing_pattern', required=True, 
                        help='s3://capk-bucket/some-hdf-pattern')
#parser.add_argument('--run-local', dest="run_local", 
#  action="store_true", default=False)
#parser.add_argument('--num-training-days', 
#  dest='num_training_days', type = int, default=16)
parser.add_argument('--start-hour', type = int, default = 3, dest="start_hour")
parser.add_argument('--end-hour', type = int, default = 7, dest="end_hour")
parser.add_argument('--num-features', type=int, default = 1, dest='num_features')

if __name__ == '__main__':
  args = parser.parse_args()
  #assert args.pattern 
  #assert len(args.pattern) > 0
  best_acc, best_param, worst_acc, worst_param, results = \
    collect_keys_and_launch(args.training_pattern, args.testing_pattern,  
     args.start_hour, args.end_hour,  args.num_features)
  print results
  print "Worst param: %s, accuracy = %s" % (worst_param, worst_acc)
  print "Best param: %s, accuracy = %s" % (best_param, best_acc)
  