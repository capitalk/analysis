import query 
import cross_info
import query_helpers
import numpy as np

MIN_DUR = None
FUTURE_HORIZON = None 

import bisect 
def does_cross_return(hdf):
  ccy = hdf.attrs['ccy']
  cross_returns = []
  bids = hdf['bid'][:]
  offers = hdf['offer'][:]
  bid_vol = hdf['bid_vol'][:]
  offer_vol = hdf['offer_vol'][:]
  min_vol = np.min([bid_vol, offer_vol], 0)
  t = hdf['t'][:]
  n = len(t)
  result = []
  for cross in cross_info.find_crossed_markets_in_hdf(hdf):
    if cross.dur >= MIN_DUR:
      i = cross.end_idx
      end_bid = bids[i]
      end_offer = offers[i]
      future_idx = bisect.bisect_right(t, t[i] + FUTURE_HORIZON, i, n)
    
      future_vol_ok = min_vol[i:future_idx] >= cross.min_vol
      future_bid_ok = bids[i:future_idx] >= cross.bid
      future_offer_ok = offers[i:future_idx] >= cross.offer
      
      # CAVEAT: I'm assuming that if the cross stuck around for less than 50ms
      # we might have missed both sides 
      if cross.dur > 50 and end_bid != cross.bid and end_offer == cross.offer:
        # bid uncrossed first
        returns = np.sum(future_bid_ok & future_vol_ok)
      # offer uncrossed first
      elif cross.dur > 50 and end_bid == cross.bid and end_offer != cross.offer:
        returns = np.sum(future_offer_ok & future_vol_ok)
      # both bid and offer changed 
      else:
        # either the cross was shorter than 50ms or both prices changed 
        # at the end 
        returns =  np.sum(future_bid_ok & future_offer_ok & future_vol_ok)
      print cross, "returns", returns, "times in", FUTURE_HORIZON / 1000.0, "seconds"
      result.append(returns > 0)
  return ccy, result
   
def combine(all_amts, (ccy,amts)):
  all_amts.setdefault(ccy, []).extend(amts)


from argparse import ArgumentParser 
parser = ArgumentParser(description='Process some integers.')
parser.add_argument('pattern', metavar='P', type=str,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--min-duration', dest='min_dur', type=int, default=None, 
  help  = 'ignore crosses which last shorter than this min. duration in milliseconds')
parser.add_argument('--future', dest = 'future', type=int, default=10000)
  
if __name__ == '__main__':
  args = parser.parse_args()
  print "Args", args
  assert args.pattern 
  assert len(args.pattern) > 0
  MIN_DUR = args.min_dur
  FUTURE_HORIZON = args.future
  df = query.run(args.pattern, 
    map_hdf = does_cross_return, 
   init = {}, combine = combine, 
   post_process = query_helpers.summarize_bool)
  print df