import query 
import numpy as np 

def map(hdf):
  """Takes an HDF, returns a tuple of min/max midprices"""
  # Note: I slice into the HDF's column to pull out a numpy array 
  midprice = (hdf['bid'][:]+hdf['offer'][:])/2.0
  return np.min(midprice), np.max(midprice)

def combine(values, (curr_min, curr_max)):
  values.append(curr_min)
  values.append(curr_max)


from argparse import ArgumentParser 
parser = ArgumentParser(description='min/max for ccy pair')
parser.add_argument('--ccy', dest='ccy', type=str, required=True, help="e.g. USDJPY")
parser.add_argument('--bucket', dest='bucekt', type=str, default='s3://capk-fxcm-hdf')
if __name__ == '__main__':
  args = parser.parse_args() 
  pattern = '*' + args.ccy + '*.hdf'
  values = query.run(args.bucket, pattern, 
    map_hdf = map, combine = combine, init = [])
  print "Min: %s, Max: %s" % (np.min(values), np.max(values))