import numpy as np
import query 
import query_helpers

# I never got around to making an easy way to pass arguments into the 
# query's mapping function so I just make them globals and set them at 
# the bottom of the file. It's ugly but it works! 
START_HOUR = None
END_HOUR = None 
RELATIVE_PRICE_CHANGE = None
TIME_OFFSET_TICKS = None

def mapper(hdf):
  """Takes an HDF, extracts its midprice for the user-specified time range, 
     filters the set of timesteps where the change in price is the same 
     sign as PRICE_CHANGE and a magnitude >= abs(PRICE_CHANGE). 
     Then further filters how often this was followed by a second move of the
     same magnitude. 
     Returns the ccy, the count for 'past' moves, and 'future' moves. These
     will later be aggregated by the function 'combine' into probabilities. 
  """ 
  # Every HDF has a set of data columns (i.e. 'bid', 'offer', etc..) and also
  # a special dictionary called 'attrs' which stores metadata. Our feature 
  # extractor stores the HDF's currency pair as a tuple called 'ccy'. 
  # Other fields you might find useful: 
  #   'year', 'month', 'day', 'start_time', 'end_time'.
  # To see how this metadata gets created see the function dict_to_hdf in hdf.py
  ccy = hdf.attrs['ccy']
  
  # if the times contained in this file fall outside our range, just return None
  # but be careful to handle None as a value when trying to combine all the 
  # mappers' returned values.
  if query_helpers.outside_hour_range(hdf, START_HOUR, END_HOUR):
    return None
  
  start_idx, end_idx = \
    query_helpers.compute_hour_indices(hdf, START_HOUR, END_HOUR)
  
  # NB: When you pull a column out of an HDF you should always slice into it, 
  # even if you just apply the identity slice [:]. This is because slicing 
  # turns the column into a numpy array, before that it's some weird HDF
  # data structure to which you can't apply math operators. 
  bid = hdf['bid'][start_idx:end_idx]
  offer = hdf['offer'][start_idx:end_idx]
  midprice = (bid+offer)/2.0
  
  past = midprice[2*TIME_OFFSET_TICKS:]
  present = midprice[TIME_OFFSET_TICKS:-TIME_OFFSET_TICKS]
  future = midprice[:-2*TIME_OFFSET_TICKS]
  
  # how much did midprice change from t-2k til t-k
  present_change_prct = (present-past) / past
  
  # how much did midprice change from t-k til t
  future_change_prct = (future-present) / present
  
  # this is the predicate we are conditioning on: was the change 
  # from the past to the present greater than some threshold
  present_indicator = \
    (np.sign(present_change_prct) == np.sign(RELATIVE_PRICE_CHANGE)) & \
    (np.abs(present_change_prct) >= np.abs(RELATIVE_PRICE_CHANGE))
  
  # this is the quantity whose probability we're trying to estimate
  # how likely is the future change also to be at least epsilon in size?
  # Note the second Logical And, which we perform since 
  # P(X | Y) = P(X and Y) * P(Y)
  # and we're going to estimate that second quantity by taking
  # Sum(X and Y) / Sum(Y)
  future_indicator = \
    (np.sign(future_change_prct) == np.sign(RELATIVE_PRICE_CHANGE)) & \
    (np.abs(future_change_prct) >= np.abs(RELATIVE_PRICE_CHANGE)) &  \
    present_indicator
  
  return ccy, np.sum(future_indicator), np.sum(present_indicator)
  
# We might be evaluating probabilities over multiple currency pairs
def combine(all_ccys, mapper_result):
  # check whether the file was skipped
  if mapper_result is not None:
    (ccy,  event_count, total) = mapper_result
    # in case we haven't seen this currency before set the default count to 0
    old_event_count, old_total = all_ccys.get(ccy, (0,0))
    all_ccys[ccy] = (old_event_count + event_count, old_total + total)


def compute_probs(all_ccys):
  """Once all the counts have been collected, do the simple task of 
     computing probs by dividing event counts by the totals
  """
  result = {}
  for (ccy,  (event_count, total)) in all_ccys.items():
    if total == 0:
      result[ccy] = 0.0
    else: 
      result[ccy]  = float(event_count) / total
  return result 
  
from argparse import ArgumentParser 
parser = ArgumentParser(
  description='P(%\ change from t to t+k >= p | p-sized move from k ms ago)')
parser.add_argument('pattern', metavar='P', type=str, required=True,
                       help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--time-offset', '-k', dest='k', type=int, required = True, 
  help = "time in seconds")
parser.add_argument('--price-change', '-p', dest='p', type=float, required=True, 
  help = 'price change in percent pips, i.e. 2 means future = current * 1.0002')
parser.add_argument('--start-hour', dest='start_time', type=int, default = 0)
parser.add_argument('--end-hour', dest='end_hour', type=int, default = 24)
if __name__ == '__main__':
  # parse args and then set globals to the values of these args, 
  # since we don't currently have any way of passing args into mappers

  args = parser.parse_args()  
  RELATIVE_PRICE_CHANGE = args.price_change * 10**-4
  TIME_OFFSET_TICKS = args.time_offset / 10

  START_HOUR = args.start_hour 
  END_HOUR = args.end_hour 
  assert START_HOUR < END_HOUR
  
  print query.run(args.pattern, 
    map_hdf = mapper, 
   init = {}, combine = combine, 
   post_process = compute_probs)
  