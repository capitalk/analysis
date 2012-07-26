from optparse import OptionParser
import numpy as np
import query 
import pandas

def cross_durations(hdf):
  ccy = hdf.attrs['ccy']
  bids = hdf['bid'][:]
  offers = hdf['offer'][:]
  t = hdf['t'][:]
  n = len(t)
  crossed = bids > offers 
  print "%d timesteps are crossed" % np.sum(crossed) 
  durs = []
  last_idx = 0
  for idx in np.where(crossed)[0]:
    if last_idx > idx:
      print "Skipping", idx, "at t =", t[idx]
    else:
      start_bid = bids[idx]
      start_offer = offers[idx]
      start_t = t[idx]
      last_idx = idx + 1
      while bids[last_idx] == start_bid and offers[last_idx] == start_offer and last_idx <= n:
        last_idx += 1
      # very rare possibility that there's a cross at the end of the file
      # which never becomes uncrossed
      if last_idx == n: 
        print "Hit the end of the day!"
        break 
      else:
        dur = t[last_idx] - start_t
        durs.append(dur)
  print "After de-duplication, only", len(durs), "crosses are left..."
  return ccy, durs   
    
def combine(all_durs, (ccy,durs)):
  if ccy in all_durs: all_durs[ccy].extend(durs)
  else: all_durs[ccy] = durs
  return all_durs

def convert_to_dataframe(all_durs):
  cols = { 
    'min': [np.min(durs) for durs in all_durs.values()], 
    'max':  [np.max(durs) for durs in all_durs.values()], 
    'median': [np.median(durs) for durs in all_durs.values()] , 
    'count': [len(durs) for durs in all_durs.values()],
  }
  return pandas.DataFrame(cols, index = all_durs.keys())
  
parser = OptionParser(usage = "usage: %prog s3://bucket-name/key-pattern")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  assert len(args)==1
  df = query.run(args[0], map_hdf = cross_durations, 
   init = {}, combine = combine, post_process = convert_to_dataframe)
  print df
