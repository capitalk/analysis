from optparse import OptionParser
import numpy as np
import query 
import pandas

def cross_amounts(hdf):
  ccy = hdf.attrs['ccy']
  bids = hdf['bid'][:]
  offers = hdf['offer'][:]
  t = hdf['t'][:]
  n = len(t)
  crossed = bids > offers 
  print "%d timesteps are crossed" % np.sum(crossed) 
  amts = []
  last_idx = 0
  for idx in np.where(crossed)[0]:
    if last_idx > idx:
      print "Skipping", idx, "at t =", t[idx]
    else:
      start_bid = bids[idx]
      start_offer = offers[idx]
      
      last_idx = idx + 1
      
      while bids[last_idx] == start_bid and offers[last_idx] == start_offer and last_idx <= n:
        last_idx += 1
      # very rare possibility that there's a cross at the end of the file
      # which never becomes uncrossed
      if last_idx == n: 
        print "Hit the end of the day!"
        break 
      else:
        amts.append(start_bid - start_offer)
  print "After de-duplication, only", len(amts), "crosses are left..."
  return ccy, amts   
    
def combine(all_amts, (ccy,amts)):
  if ccy in all_amts: all_amts[ccy].extend(amts)
  else: all_amts[ccy] = amts
  return all_amts

def convert_to_dataframe(all_amts):
  cols = { 
    'min': [np.min(amts) for amts in all_amts.values()], 
    'max':  [np.max(amts) for amts in all_amts.values()], 
    'median': [np.median(amts) for amts in all_amts.values()] , 
    'count': [len(amts) for amts in all_amts.values()],
  }
  return pandas.DataFrame(cols, index = all_amts.keys())
  
parser = OptionParser(usage = "usage: %prog s3://bucket-name/key-pattern")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  assert len(args)==1
  df = query.run(args[0], map_hdf = cross_amounts, 
   init = {}, combine = combine, post_process = convert_to_dataframe)
  print df
