import numpy as np
import pandas
import scipy 
import scipy.weave


def mad(x):
  """mean absolute deviation from the sample median"""
  return np.mean(np.abs(x - np.median(x)))

def crossing_rate(x):
  """# of times the series crosses its own initial value / length of series"""
  init = x[0]
  gt = x > init
  lt = x <= init
  moves_down = gt[:-1] & lt[1:]
  moves_up = lt[:-1] & gt[1:]
  n_crosses = np.sum(moves_down) + np.sum(moves_up)
  return n_crosses / float(len(x))

def rolling_crossing_rate(x, w):
  code = """
    int nx = Nx[0];
    double dw = (double) w; 
    double start_val = 0;
    double prev = 0;
    double curr = 0; 
   
    for (int i = 0; i <= nx -w; ++i) {
      int num_crosses = 0;
      start_val = x[i];
      for (int j = i+1; j < i+w; ++j) {
        prev = x[j-1];
        curr = x[j];
        if ( (prev <= start_val && curr > start_val) 
             || (prev > start_val && curr <= start_val)) {
          num_crosses++; 
        }
      }
      result[i+w-1] = num_crosses / dw;
    }
    """
  result = np.zeros(len(x))
  scipy.weave.inline(code, ['x', 'w', 'result'], verbose=2)
  return result

#def rolling_crossing_rate2(x, w):
#  n = len(x)
#  result = np.zeros(n)
#  stop = n - w
#  gt = np.zeros(w, dtype='bool')
#  lte = np.zeros(w, dtype='bool')
#  up = np.zeros(w-1, dtype='bool')
#  down = np.zeros(w-1, dtype='bool')
  
#  for (i, xi) in enumerate(x[:stop]):
#    idx = i + w
#    window = x[i:idx]
#    np.greater_equal(window, xi, out=gt)
#    np.less(window, xi, out=lte)
#    np.logical_and(gt[1:], lte[:-1], out=up)
#    np.logical_and(gt[:-1], lte[1:], out=down)
#    result[idx] = (np.sum(up) + np.sum(down))
#  result /= float(w)
#  return result 
  
def rolling_var(x, w):
  x = np.asarray(x)
  assert isinstance(w, int)
  means = pandas.rolling_mean(x, w)
  code = """
    int nx = Nx[0];
    double dw = (double) w; 

    for (int i = 0; i <= nx -w; ++i) {
      int idx = i+w-1; 
      double mean = means[idx];  
      double var_sum = 0.0;
      for (int j = i; j < i+w; ++j) {
        double diff = x[j] - mean;
        var_sum += diff * diff; 
      }
      result[idx] = var_sum / dw;
  }
    """
  n = len(x)
  result = np.zeros(n)
  if n > w:
    scipy.weave.inline(code, ['x', 'w', 'means', 'result'], verbose=2)
  return result

def rolling_std(x, w):
  return np.sqrt(rolling_var(x,w))
  
def rolling_fn(x, w, fn):
  #print "Applying rolling fn %s with window size %d" % (fn, w)
  builtin = {
    np.mean: pandas.rolling_mean, 
    np.median: pandas.rolling_median, 
    np.min: pandas.rolling_min, 
    np.max: pandas.rolling_max, 
    np.var: rolling_var, # not sure why I get NaN from pandas functions 
    np.std: rolling_std, 
    crossing_rate: rolling_crossing_rate, 
  }.get(fn, None)
  if builtin:
    aggregated = builtin(x, w)
  elif fn == mad:
    medians = pandas.rolling_median(x, w)
    abs_diffs = np.abs(x - medians)
    aggregated = pandas.rolling_mean(abs_diffs, w)
  else:
    aggregated = pandas.rolling_apply(x, w, fn)
  n_bad = np.sum(~np.isfinite(aggregated[w:]))
  if n_bad > 0:
    print "[rolling_fn] Number bad entries:", n_bad
  return aggregated