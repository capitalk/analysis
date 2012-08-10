import numpy as np
#import sklearn.ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
import math 
import cloud
import cloud_helpers 

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
    
    'aggregator' : [np.median], #[np.median, mad, crossing_rate],
     
     # window sizes in seconds
     'aggregator_window_size' : [1], #[1, 10, 100], 
     'normalizer': [LaplaceScore], #[None, ZScore, LaplaceScore],

     # all times in seconds-- if the time isn't None then we measure the 
     # prct change relative to that past point in time
     'past_lag':  [None], #[None, 5, 20, 30, 40, 60, 120, 600],
    
     'transform' : [np.square], #[None, np.square]
  }
  def filter(x):
    return \
      (x.past_lag and x.past_lag < x.aggregator_window_size) or \
      (x.normalizer is not None and x.past_lag is not None)
  return all_param_combinations(options, filter = filter)

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
  
  lag = param.past_lag
  if not lag:
    lagged = aggregated
  
  else:
    # like window size, lag is in seconds, but ticks are 100ms
    future = aggregated[10*lag:]
    present  = aggregated[:-10*lag]
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
  
  for (i, f) in enumerate(features):
    if f.normalizer is None:
      normalizers.append(None)
    else:
        # only use first half of dataset for estimating normalizer params
        # so we don't overfit by normalization
      data = []
      for hdf in hdfs[:half]:
        print "[fit_normalizers]", i, f, hdf
        col = hdf[f.raw_feature][:]
        data.append(col)
      data = np.concatenate(data)
      N = f.normalizer()
      N.fit(data)
      normalizers.append(N)
  return normalizers 
  
def construct_dataset(hdfs, features, future_offset, start_hour, end_hour, normalizers = None):
  inputs = []
  outputs = []
  # future offset is in seconds, ticks are 100ms
  future_offset_ticks = 10 * future_offset
  print "[construct_dataset] Future_offset = %s" % future_offset_ticks
  
  # TODO: Actually use start_hour and end_hour 
  if normalizers is None:
    normalizers = fit_normalizers(hdfs, features)
  
  for hdf in hdfs:
    
    bids = hdf['bid'][:]
    # signal is: will the bid go up in some number of seconds
    y = bids[future_offset_ticks:] > bids[:-future_offset_ticks]
    outputs.append(y)
     
    cols = []
    # construct all the columns for a subset of rows
    for (i, f) in enumerate(features):
      print "[construct_dataset]", i, f, hdf 
      N = normalizers[i]
      col, _ = extract_feature(hdf, f, N)
      #print "A", col
      
      # remove last future_offset*10 ticks, since we have no output for them
      col = col[:-future_offset_ticks]
      #print "B", col
      cols.append(col)
    mat = np.hstack(cols)
    # print "mat shape", mat.shape
    inputs.append(mat)
  return np.vstack(inputs).T, np.concatenate(outputs), normalizers
  
def download_hdfs(bucket, keys):
  print "[download_hdfs]", keys
  hdfs = []
  for key in keys:
    hdf = cloud_helpers.download_hdf_from_s3(bucket, key)
    hdfs.append(hdf)
  print "[download_hdfs] Done!"
  return hdfs 

# affects only future self, no time travel but watch for self improvement
# and itching, but actually it just might be autoimmune or a mosquito 
# It's often hard to tell the difference. 
# To shed further light on this situation Timir should probably hire Sarah. 
# ...to do archival research for him. And a little bit of divination and/or
# creative writing. 
def eval_new_param(bucket, hdf_keys, old_params, new_param, 
    num_training_days = 16, start_hour = 3, end_hour = 7, future_offset = 30):
  if new_param in old_params:
    return None
  params = old_params + [new_param]
  n_files = len(hdf_keys)
  
  accs = []
  for test_idx in np.arange(n_files)[num_training_days:]:
    training_filenames = hdf_keys[(test_idx - num_training_days):test_idx]
    test_filename = hdf_keys[test_idx]
    print "Downloading HDF files for %d / %d" % (test_idx, n_files)
    training_hdfs = download_hdfs(bucket, training_filenames)
    test_hdf = cloud_helpers.download_hdf_from_s3(bucket, test_filename)
    print "Constructing dataset for %d / %d" % (test_idx, n_files)
    x_train, y_train, normalizers = \
      construct_dataset(training_hdfs, params, future_offset, start_hour, end_hour, None)
    x_test, y_test, _ = \
      construct_dataset([test_hdf], params, future_offset, start_hour, end_hour, normalizers)
    print "x_train shape: %s, y_train shape: %s" % (x_train.shape, y_train.shape)
    print "Training model..."
    #model = RandomForestClassifier(n_estimators = 100)
    #model = LinearSVC()
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print "Generating predictions"
    pred = model.predict(x_test)
    acc = np.mean(pred == y_test)
    print "Accuracy for %d / %d = %s" % (test_idx, n_files, acc)
    accs.append(acc)
  med_acc = np.median(acc)
  print "Median accuracy: %s" % med_acc
  return med_acc
    
def launch_jobs(hdf_bucket, hdf_keys, raw_features = None, 
    num_training_days = 16, start_hour = 3, end_hour = 7):
  if raw_features is None:
    print "Downloading ", hdf_keys[0]
    hdf = cloud_helpers.download_hdf_from_s3(hdf_bucket, hdf_keys[0])
    raw_features = hdf.attrs['features']
  print "Raw features", raw_features
  all_params = gen_feature_params(raw_features)
  print "Launching %d jobs" % len(all_params)
  old_chosen_params = []
  label = 'Evaluating %d features' % len(all_params)
  def do_work(p):
    return eval_new_param(hdf_bucket, hdf_keys, old_chosen_params, p, 
      num_training_days = num_training_days, 
      start_hour = start_hour, 
      end_hour = end_hour)
  jids = cloud.map(do_work, 
    all_params, 
    _env = 'compute', 
    _label=label, 
    _type = 'f2')
  worst_acc = 1
  worst_param = None 
  best_acc = 0
  best_param = None
  results = {}
  for (i, acc) in enumerate(cloud.iresult(jids)):
    results[all_params[i]]  = acc
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
     
def single_feature_search(pattern, num_training_days, start_hour, end_hour):
  bucket, key_pattern = cloud_helpers.parse_bucket_and_pattern(pattern)
  if len(key_pattern) == 0:
    key_pattern = '*'
  key_names = cloud_helpers.get_matching_key_names(bucket, key_pattern)
  #raw_features = get_common_features(bucket, key_names)
  #print "Raw features:", raw_features
  return launch_jobs(bucket, key_names, raw_features = None, 
    training_window = num_training_days, 
    start_hour = start_hour, 
    end_hour = end_hour)
  

from argparse import ArgumentParser 
parser = ArgumentParser(description='Look for single best feature')
parser.add_argument('pattern', metavar='P', type=str,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--cloud-simulator', dest="cloud_simulator", 
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
  if args.cloud_simulator:
    cloud.start_simulator()
  best_acc, best_param, worst_acc, worst_param, results = \
    single_feature_search(args.pattern, args.num_training_days, 
    args.start_hour, args.end_hour)
  print results
  print "Worst param: %s, accuracy = %s" % (worst_param, worst_acc)
  print "Best param: %s, accuracy = %s" % (best_param, best_acc)
  