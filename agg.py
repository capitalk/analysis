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
  
def rolling_fn(x, w, fn):
  builtin = {
    np.mean:pandas.rolling_mean, 
    np.median:pandas.rolling_median, 
    np.min:pandas.rolling_min, 
    np.max:pandas.rolling_max, 
    np.var:pandas.rolling_var, 
    np.std: pandas.rolling_std, 
  }.get(fn, None)
  if builtin:
    aggregated = builtin(x, w, min_periods = w)
  elif fn == mad:
    medians = pandas.rolling_median(x, w)
    abs_diffs = np.abs(x - medians)
    aggregated = pandas.rolling_mean(abs_diffs, w)
  elif fn == crossing_rate:
    aggregated = rolling_crossing_rate(x,w)
  else:
    aggregated = pandas.rolling_apply(x, w, fn)
  return aggregated[w:]