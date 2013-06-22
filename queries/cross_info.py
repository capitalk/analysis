import numpy as np
import collections

Cross = collections.namedtuple('Cross',
  ( 'start_idx', 'end_idx',
    'start_t', 'end_t', 'dur',
    'bid', 'offer', 'amt',
    'bid_vol', 'offer_vol', 'min_vol'))
    

def find_crossed_markets_in_dataframe(d):
  bids = d['bid']
  bid_vols = d['bid_vol']
  offers = d['offer']
  offer_vols = d['offer_vol']
  t = d['t']
  n = len(t)
  crossed_mask = bids > offers
  crossed_indices =  np.where(crossed_mask)[0]
  print "%d timesteps are crossed" % len(crossed_indices)
  search_idx = 0
  crosses = []
  for start_idx in crossed_indices:
    if search_idx > start_idx:
      print "Skipping", start_idx, "at t =", t[start_idx]
    else:
      bid = bids[start_idx]
      offer = offers[start_idx]
      bid_vol = bid_vols[start_idx]
      offer_vol = offer_vols[start_idx]
      start_t = t[start_idx]
      
      search_idx = start_idx + 1
      while bids[search_idx] == bid and \
        offers[search_idx] == offer and \
          search_idx <= n:
        # take the min volume across the whole time market is crossed
        # just so we get pessimistic estimates
        bid_vol = min(bid_vol, bid_vols[search_idx])
        offer_vol = min(offer_vol, offer_vols[search_idx])
        search_idx += 1    
      # very rare possibility that there's a cross at the end of the file
      # which never becomes uncrossed
      if search_idx == n:
        print "Hit the end of the day!"
        break
      else:
        end_t = t[search_idx]
        cross = Cross(
          start_idx = start_idx, 
          end_idx = search_idx,
          start_t = start_t,
          end_t = end_t, 
          dur = end_t - start_t,
          bid = bid, 
          offer = offer, 
          amt = bid - offer,
          bid_vol = bid_vol, 
          offer_vol = offer_vol,
          min_vol = min(bid_vol, offer_vol)
        )
        crosses.append(cross)
  print "After de-duplication, ", len(crosses), "crosses are left..."
  return crosses
  
def find_crossed_markets_in_hdf(hdf):
  d = { 
    't': hdf['t'][:], 'bid':hdf['bid'][:], 'offer':hdf['offer'][:], 
    'bid_vol': hdf['bid_vol'][:], 'offer_vol':hdf['offer_vol'][:]
  }
  return find_crossed_markets_in_dataframe(d)