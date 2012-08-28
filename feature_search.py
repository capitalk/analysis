import numpy as np
#import sklearn.ensembles
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline

from multiprocessing import Pool 

from column_normalizer import ColumnNormalizer
import math 
import cloud
import cloud_helpers 
import h5py 
from agg import mad, crossing_rate, rolling_fn 

from normalization import ZScore 
import pylab
import bisect 
import pandas 


def profitable_future_change(bid, offer, offset):
  n = len(bid) - offset
  result = np.zeros(n)
  buy_profit = bid[offset:] - offer[:-offset]
  buy_idx = buy_profit > 0 

  # buying is profitable when the price goes up later 
  result[buy_idx] = buy_profit[buy_idx]  
  sell_profit = bid[:-offset] - offer[offset:]
  sell_idx = sell_profit > 0
  # selling is profitable when the price goes down later 
  result[sell_idx] = -sell_profit[sell_idx]
  return result 

def spread_normalized_future_change(bid, offer, offset):
  spread = offer - bid 
  smoothed_spread = pandas.rolling_mean(spread, offset)
  smoothed_spread = smoothed_spread[offset:]
  profitable_change = profitable_future_change(bid, offer, offset)
  return profitable_change / smoothed_spread 

def spread_normalized_midprice_change(bid, offer, dt):
  spread = bid - offer
  smoothed_spread = pandas.rolling_mean(spread, dt)[dt:]
  midprice = (bid+offer)/2.0
  change = midprice[dt:] - midprice[:-dt]
  return change / smoothed_spread #pandas.rolling_median(change, dt)[dt:] / smoothed_spread[dt:]

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

Result = namedtuple('Result', 
  ('neg', 'zero', 'pos',  'accuracy', 'precision', 'recall', 'score'))
  
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
    'aggregator' : [np.mean, np.std], #, crossing_rate],
     
     # window sizes in seconds
     'aggregator_window_size' : [100, 1000], 
     'normalizer': [None], #ZScore], # [ZScore, LaplaceScore]
     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None, 300, 600, 1200, 2400, 4800], 
     'transform' : [None], 
  }
  def filter(x):
    return (x.past_lag and x.past_lag < x.aggregator_window_size) or \
     (x.aggregator is None and x.aggregator_window_size is not None) or \
     (x.aggregator is not None and x.aggregator_window_size is None)
  return all_param_combinations(options, filter = filter)

def compute_indices_from_hours(hdf, start_hour = None, end_hour = None):
  milliseconds_per_hour = 60 * 60 * 1000 

  if start_hour is None:
    start_millisecond = 0
  else:
    start_millisecond = start_hour * milliseconds_per_hour
    
  if end_hour is None:
    end_millisecond = 24 * milliseconds_per_hour 
  else: 
    end_millisecond = end_hour * milliseconds_per_hour
  
  t = hdf['t'][:]
  file_ok = start_millisecond < t[-1] and end_millisecond > t[0]
  if file_ok:
    start_idx = bisect.bisect_right(t, start_millisecond)
    end_idx = bisect.bisect_left(t, end_millisecond) 
    return True, start_idx, end_idx
  else:
    return False, None, None 
  
def construct_dataset(hdfs, features, future_offset, 
     start_hour = 3, end_hour = 7):
  inputs = []

  all_lags =[(f.past_lag if f.past_lag else 0) for f in features] 
  max_lag = max(all_lags)
  
  all_aggregator_window_sizes = \
     [(f.aggregator_window_size if f.aggregator_window_size else 0) \
      for f in features]
   
  max_aggregator_window_size = max(all_aggregator_window_sizes)
  for  hdf in hdfs:
    file_ok, start_idx, end_idx = compute_indices_from_hours(hdf, start_hour, end_hour)
    if not file_ok or (end_idx - start_idx) <= future_offset:
      print "Skipping %s since it doesn't contain hours %d-%d" % \
        (hdf.filename, start_hour, end_hour)
      continue
    
    cols = []
    # construct all the columns for a subset of rows
    for param in features:
      #print "--", param
      """
      Get the raw feature from the HDF and then:
        (1) apply the rolling aggregator over the raw data
        (2) normalize the data
        (3) optionally transform the data
        (4) optionally get the percent change from some point in the past
      """
      raw_feature = param.raw_feature
      x = hdf[raw_feature][start_idx:end_idx]
      assert np.all(np.isfinite(x))
      assert len(x) > future_offset
      if 'vol' in raw_feature:
        x /= 10.0 ** 6
      elif raw_feature == 't':
        x /= float(x[-1])
      assert np.all(np.isfinite(x)), "Raw features %s contains bad data" % raw_feature
      #n = len(x)
      if param.aggregator:
        w = param.aggregator_window_size
        x = rolling_fn(x, w, param.aggregator)[w:]
        assert np.all(np.isfinite(x)), \
          "Got bad data from rolling aggregator %s (win size = %d)" %\
             (param.aggregator, param.aggregator_window_size) 
      if param.transform:
        x = param.transform(x)
      assert np.all(np.isfinite(x))
      lag = param.past_lag
      if lag:
        past = x[:-lag]
        present = x[lag:]
        x = (present - past)
    
      # remove last future_offset ticks, since we have no output for them
      x = x[:-future_offset]
      
      # skip some of the past if it's also seen by the feature with max. lag
      
      amt_lag_trim = max_lag - (0 if param.past_lag is None else param.past_lag)
      #print max_lag, param.past_lag, amt_lag_trim
      x = x[amt_lag_trim:]
      # if you're being aggregated in smaller windows than some other feature
      # then you should snip off some of your starting samples 
      amt_agg_window_trim = max_aggregator_window_size - \
        (0 if param.aggregator_window_size is None else param.aggregator_window_size)
      #print max_aggregator_window_size, param.aggregator_window_size, amt_agg_window_trim
      x = x[amt_agg_window_trim:]
      cols.append(x)
    shapes = [c.shape for c in cols]
    #print "shapes", shapes
    assert all([s == shapes[0] for s in shapes]), \
      "Not all shapes the same: %s" % shapes
    mat = np.array(cols) 
    #print "Final shape: ", mat.shape
    assert np.all(np.isfinite(mat))
    inputs.append(mat)
  total_lag = max_aggregator_window_size + max_lag
  X = np.hstack(inputs).T
  print "Final shape", X.shape
  return X, total_lag 

def construct_outputs(hdfs, future_offset, lag = 0, start_hour = None, end_hour = None):
  outputs = []
  for hdf in hdfs:
    file_ok, start_idx, end_idx = compute_indices_from_hours(hdf, start_hour, end_hour)
    if not file_ok or (end_idx - start_idx) <= future_offset:
      continue
    # signal is: will the bid go up in some number of seconds
    bids = hdf['bid'][start_idx:end_idx]
    offers = hdf['offer'][start_idx:end_idx]
    
    y = np.zeros(len(bids) - future_offset, dtype='int')
    future_bids = bids[future_offset:]
    past_bids = bids[:-future_offset]
    assert len(future_bids) == len(past_bids), \
      "Expected future_bids (len %d) and past_bids (len %d) to be same length (future_offset = %d)" % \
      (len(future_bids), len(past_bids), future_offset)
 
    future_offers = offers[future_offset:]
    past_offers = offers[:-future_offset]
    assert len(future_offers) == len(past_offers), \
      "Expected future_offers (len %d) and past_offers (len %d) to be same length (future_offset = %d)" % \
      (len(future_offers), len(past_offers), future_offset)

    y[future_bids > past_offers] = 1
    y[future_offers < past_bids] = -1

    #y[offers[future_offset:] < bids[:-future_offset]] = -1 
    #y = spread_normalized_midprice_change(bids, offers, future_offset)

    assert np.all(np.isfinite(y))
    y = y[lag:]
    outputs.append(y)
  return np.concatenate(outputs)
     
def common_features(hdfs):
  feature_set = None 
  assert len(hdfs) > 0
  for hdf in hdfs:
    curr_set = set(hdf.attrs['features'])
    if feature_set is None:
      feature_set = curr_set
    else:
      feature_set = feature_set.intersection(curr_set)
  assert feature_set is not None
  for feature_name in list(feature_set):
    vec = hdf[feature_name]
    if feature_name == 't':
      print "Skipping time feature"
      feature_set.remove(feature_name)
    elif np.all(vec == vec[0]):
      print "Skipping feature %s since it's constant %s" % (feature_name, vec[0])
      feature_set.remove(feature_name)
    
  return feature_set 

def combine_precision_recall(p, r, min_r = 0.001):
  return p * min(1.0, r / min_r) 
  
def prediction_score(y, pred):
  # if the outputs were discrete we could just 
  # do something simple like 'correct = (y == pred)'
  # but we also want to gracefully handle 
  # spread-normalized continuous predictions where
  # any values >= 1 are up predictions, <= -1 are down
  # predictions and everything else is considered 
  # equivalently neutral 
 
  pred_up = pred >= 1
  pred_down = pred <= -1
  pred_nz = pred_up | pred_down
  pred_z = ~pred_nz 

  y_up = y >= 1
  y_down = y <= -1
  y_nz = y_up | y_down
  y_z = ~y_nz

  correct_up = y_up & pred_up
  correct_down = y_down & pred_down
  correct_nz = correct_up | correct_down
  correct_z = y_z & pred_z
  correct = correct_nz | correct_z   

  num_correct_nz = np.sum(correct_nz) 
  precision_denom = float(np.sum(pred_nz))
  if precision_denom > 0:
    precision = num_correct_nz / precision_denom
  else:
    precision = 0
  recall_denom = float(np.sum(y_nz))
  if recall_denom > 0:
    recall = num_correct_nz / recall_denom
  else:
    recall = 0
  score = combine_precision_recall(precision, recall)
  
  n_neg = np.sum(pred_down)
  n_zero = np.sum(pred_z)
  n_pos =  np.sum(pred_up) 
  print "pred  zero = %d, neg = %d, pos = %d" % \
    (n_zero, n_neg , n_pos )
  print "precision =", precision 
  print "recall =", recall 
  print "score =", score
  result = Result(zero = n_zero, neg = n_neg, pos = n_pos, 
    accuracy = np.mean(correct),
    precision = precision, recall = recall, score = score, 
  )
  return result 

def cross_validate_model(model, x, y, folds = 5, parallel=True):
  n = len(y)
  kf = KFold(n, folds)
  results = []
  def train_fold(train_index, test_index):
    x_train, x_test = x[train_index, :], x[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    if np.all(np.isfinite(x_test)) and np.all(np.isfinite(x_train)):
      model.fit(x_train, y_train)
      pred = model.predict(x_test)
      prob = model.predict_proba(x_test)
      uncertain = np.max(prob, 0) < 0.5
      pred[uncertain] = 0
      result = prediction_score(y_test, pred)
    else:
      result = None
    results.append(result)

  for (train_index, test_index) in kf:
    train_fold(train_index, test_index)
  #y_test = np.concatenate(y_tests)
  #y_pred = np.concatenate(y_preds)
  #return prediction_score(y_test, y_pred) 
  if None in results:
    combined_result = None
  else:
    combined_result = Result(
      zero = np.mean([r.zero for r in results]),
      neg = np.mean([r.neg for r in results]), 
      pos = np.mean([r.pos for r in results]),
      precision = np.mean([r.precision for r in results]), 
      recall = np.mean([r.recall for r in results]), 
      accuracy = np.mean([r.accuracy for r in results]),
      score = np.mean([r.score for r in results]), 
    )
  return combined_result 
def eval_params(training_hdfs, testing_hdfs, old_params, new_param, start_hour, end_hour, future_offset):
  if new_param.raw_feature is None:
    raw_features = common_features(training_hdfs)
  elif isinstance(new_param.raw_feature, list):
    raw_features = new_param.raw_feature
  else:
    raw_features = [new_param.raw_feature]
  print "Raw features: ", raw_features
  results = {}
  y_train = None
  y_test = None
  best_score = 0  
  best_model = None
  print "old_params", old_params
  print "new_param", new_param
  print "raw_features", raw_features 
  for (i, raw_feature) in enumerate(raw_features):
    param = copy_params(new_param, raw_feature = raw_feature)
    print param
    if param in old_params:
      print "...duplicate param in dataset, skipping..."
      results[param] = None
    elif param.raw_feature == 't' and param.aggregator not in [np.mean, np.median, None]:
      print "raw_feature = time with aggregator = %s, skipping..." % param.aggregator
      results[param] = None
    else:  
      params = old_params + [param]
      x_train, lag = \
        construct_dataset(training_hdfs, params, future_offset, start_hour, end_hour)
      if y_train is None:
        y_train = construct_outputs(training_hdfs, future_offset, lag, start_hour, end_hour)
        y_test = construct_outputs(testing_hdfs, future_offset, lag, start_hour, end_hour)


      # x_train, normalizers = normalize_data(x_train, params = params)
      print "x_train shape: %s, y_train shape: %s" % (x_train.shape, y_train.shape)

      # print "Training model..."
      if np.all(np.isfinite(x_train)): x_train_ok = True
      else:
        x_train_ok = False
        print "Training data contains NaN or infinity"
      if np.all(np.isfinite(y_train)): y_train_ok = True
      else:
        y_train_ok = False
        print "Training label contains NaN or infinity"
        
      if x_train_ok and y_train_ok:
        
        print "train zero = %d, neg = %d, pos = %d" % \
         (np.sum(y_train == 0), np.sum(y_train < 0), np.sum(y_train > 0))
        
        print "Scoring training data"
        #n_iter = min(2, int(math.ceil(10.0**6 / x_train.shape[0])))
        #model = SGDClassifier(loss = 'log', n_iter = n_iter, shuffle = True)
        n_train = len(y_train)
        n_features = x_train.shape[1]
         
        # model = RandomForestRegressor(n_estimators = 3, max_depth=max_depth, min_samples_leaf = min_samples_leaf)
        #model = LinearRegression()
        def mk_rf():
          min_samples_leaf = int(round(np.log2(n_train)))
          n_estimators = 7
          model = RandomForestClassifier(
            n_estimators = n_estimators, 
            max_depth = min(4, 1+n_features), 
            min_samples_leaf = min_samples_leaf, 
            max_features = n_features,
          )
          return model
        model = mk_rf()
        #model = LogisticRegression() 
        print model
        normalizer = ColumnNormalizer([p.normalizer for p in params])
        pipeline = Pipeline([ ('normalize', normalizer), ('model', model)])
        train_result = cross_validate_model(pipeline, x_train, y_train) 
        print train_result
        print "---"
        if train_result and train_result.score >= best_score:
          best_score = train_result.score
          best_model = pipeline
          x_test, _ = \
           construct_dataset(testing_hdfs, params, future_offset, start_hour, end_hour)
          # x_test = normalize_data(x_test, normalizers = normalizers)
          print "x_test shape: %s, y_test shape: %s" % (x_test.shape, y_test.shape)
          print "Scoring testing data"
          print "test  zero = %d, neg = %d, pos = %d" % \
            (np.sum(y_test == 0), np.sum(y_test < 0), np.sum(y_test > 0))
          pipeline.fit(x_train, y_train)
          test_pred = pipeline.predict(x_test)
          test_prob = pipeline.predict_proba(x_test)
          uncertain = np.max(test_prob, 0) < 0.5
          test_pred[uncertain] = 0
          assert len(test_pred) == len(y_test), "len pred = %d, len y_test = %d" % (len(test_pred), len(y_test))
          test_result = prediction_score(y_test, test_pred)
          print
          print "Current best score %s, \ntrain result %s \ntest_result %s" % \
            (best_score, train_result, test_result)
          print
          print
        else:
          test_result = None
        results[param] = (train_result, test_result)
        
        
      else:
        print "Skipping due to bad data", param 
        results[param] = None
        print 
  return best_score, best_model, results

def download_files(bucket, keys, parallel = False):
  hdfs = []

  def download(k):
    try:
      return cloud_helpers.download_file_from_s3(bucket, k)
    except:
      try:
        return cloud_helpers.download_file_from_s3(bucket, k)
      except Exception, e:
        return e

  def receive(filename):
    if isinstance(filename, str) or isinstance(filename, unicode):
      print "Downloaded", filename
      hdf = h5py.File(filename)
      hdfs.append(hdf)
    else:
      # got back an exception from a worker, raise it 
      raise filename

  if parallel:
    map(receive, cloud.mp.iresult(cloud.mp.map(download, keys)))
      
  else:
    for k in keys:
      receive(download(k))
  return hdfs
    
  
# affects only future self, no time travel but watch for self improvement
# and itching, but actually it just might be autoimmune or a mosquito 
# It's often hard to tell the difference. 
# To shed further light on this situation Timir should probably hire Sarah. 
# ...to do archival research for him. And a little bit of divination and/or
# creative writing. 
def download_and_eval(bucket, training_keys, testing_keys, old_params, new_param, 
    start_hour = 3, end_hour = 7, future_offset = 450):
 
  n_train = len(training_keys)
  n_test = len(testing_keys)
  print "Downloading %d training HDFs..." % n_train
  training_hdfs = download_files(bucket, training_keys)

  print "Downloading %d testing HDFs..." % n_test
  testing_hdfs = download_files(bucket, testing_keys)
  result = \
    eval_params(training_hdfs, testing_hdfs, old_params, new_param, start_hour, end_hour, future_offset)    
  print result
  return result
    
def launch_jobs(hdf_bucket, training_keys, testing_keys, 
    raw_features = None, start_hour = 3, end_hour = 7, 
    num_features = 1, future_offset = 600, profile = True):
  all_possible_params = gen_feature_params(raw_features)

  chosen_params = []
  def worker_wrapper(new_param):
    return download_and_eval(hdf_bucket, training_keys, testing_keys,
      chosen_params, 
      new_param,  
      start_hour = start_hour, 
      end_hour = end_hour, 
      future_offset = future_offset)
  
  for feature_num in xrange(num_features):

    print "=== Searching for feature #%d ===" % (feature_num+1)
    print "Launching %d jobs over %d training files and %d testing files" %  \
      (len(all_possible_params), len(training_keys), len(testing_keys))
    
    label = 'Evaluating %d parameter sets for feature #%d' % \
      (len(all_possible_params), feature_num+1)

    jids =\
      cloud.map(worker_wrapper, all_possible_params, 
        _env = 'compute', 
        _label=label, 
        _type = 'f2', 
        _profile = profile)
    results = {}
    best_result = None
    best_model = None
    for (i, (curr_best_score, curr_best_model, results)) in enumerate(cloud.iresult(jids)):
      if results is None:
        results = {}
      else:
        assert isinstance(results, dict)
      print "Got %d results with best score = %s" % (len(results), curr_best_score)
      for (param, r) in results.items():
        key = tuple(chosen_params + [param])
        results[key] = r
        #if r is None:
          # print param, "<skipped>"
        if r is not None:
          train_result, test_result = r
          score = train_result.score
          if best_result is None or not np.isfinite(best_result['train'].score) or \
             best_result['train'].score <  score:
            print "New best:", key
            print "result for training data:", train_result
            if test_result: 
              print "result for testing data:", test_result
            print
            best_result  = { 
              'params':key, 
              'train': train_result,  
              'test': test_result
            }
            best_model = curr_best_model 
    print 
    print "Current best for %d features: %s" % (feature_num+1, best_result)
    print
    curr_best_params = best_result['params']
    if len(curr_best_params) < feature_num+1:
      print "Got no improvement from adding %d'th feature, stopping..."
      break
    else:
      chosen_params.append(curr_best_params[-1])
  return best_result, best_model, results

     
def collect_keys_and_launch(training_pattern, testing_pattern, 
    start_hour = 3, end_hour = 7, num_features = 1, future_offset = 600,
    profile = False):
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
  assert len(training_names) > 0
  assert len(testing_names) > 0
  assert training_bucket == testing_bucket, \
    "Expected training bucket %s to be same as testing bucket %s" % (training_bucket, testing_bucket)
 
  return launch_jobs(training_bucket, training_names, testing_names, 
    raw_features = None, 
    start_hour = start_hour, 
    end_hour = end_hour, 
    num_features = num_features, 
    future_offset = future_offset, 
    profile = profile)
  

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
parser.add_argument('--future-ticks', type=int, default = 600, dest='future_ticks')
parser.add_argument('--profile', default=False, action="store_true", dest="profile")
parser.add_argument('--save-model', default=None, dest='model_filename')
if __name__ == '__main__':
  args = parser.parse_args()
  #assert args.pattern 
  #assert len(args.pattern) > 0
  best_result, best_model, all_results = \
    collect_keys_and_launch(args.training_pattern, args.testing_pattern,  
     args.start_hour, args.end_hour,  args.num_features, 
     args.future_ticks, 
     args.profile)
  print all_results
  print
  
  print 
  print "Best:", best_result
  if args.model_filename:
    print 
    print "Writing model to file..."
    f = open(args.model_filename)
    import pickle 
    pickle.dump(best_model, f)
    f.close()
    print "Done" 
