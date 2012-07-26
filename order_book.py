
from collections import namedtuple 
  
OFFER_SIDE = True
BID_SIDE = False

Order = namedtuple("Order", ("timestamp", "side", "level", "price", "size"))  

ADD_ACTION_TYPE = 'A'
DELETE_ACTION_TYPE = 'D' 
MODIFY_ACTION_TYPE = 'M'

# actions are now just tuples, so use these positions instead of fields
Action = namedtuple('Action', ('action_type', 'side', 'price', 'size'))

class OrderBookStats:
  def __init__(self, best_bid_price, best_bid_vol, best_offer_price, best_offer_vol):
    self.best_bid_price = best_bid_price
    self.best_bid_volume = best_bid_vol
    self.best_offer_price = best_offer_price
    self.best_offer_volume = best_offer_vol
    self.bid_tr8dr = 0
    self.offer_tr8dr = 0
    self.filled_bid_volume = 0
    self.filled_bid_count = 0
    self.filled_offer_volume = 0
    self.filled_offer_count = 0
    self.canceled_bid_volume = 0
    self.canceled_bid_count = 0
    self.canceled_offer_volume = 0
    self.canceled_offer_count = 0
    self.added_bid_volume = 0
    self.added_bid_count = 0
    self.added_offer_volume = 0
    self.added_offer_count = 0
    self.added_best_bid_volume = 0
    self.added_best_bid_count = 0
    self.added_best_offer_volume = 0
    self.added_best_offer_count = 0
    self.deleted_bid_volume = 0
    self.deleted_bid_count = 0
    self.deleted_offer_volume = 0
    self.deleted_offer_count = 0
 
OrderBook  = namedtuple("OrderBook", 
  ( "day", "last_update_time", "last_update_monotonic", 
    "bids", "offers", "actions"))
    
    

def compute_stats(ob): 
  # Warning: assumesa len(ob.bids) > 0 and len(ob.offers) > 0, 
  # be sure to filter out orderbooks where some side is empty 
  
  best_offer_price = ob.offers[0].price
  best_bid_price = ob.bids[0].price
  
  stats = OrderBookStats(best_bid_price, ob.bids[0].size, best_offer_price, ob.offers[0].size)
  
  for a in ob.actions: 
    action_type, side, p, v  = a
    
    if action_type == ADD_ACTION_TYPE:
      stats.offer_tr8dr += (p - best_offer_price) / best_offer_price 
      if side == OFFER_SIDE:
        stats.added_offer_volume += v
        stats.added_offer_count += 1
        if p <= best_offer_price:
          stats.added_best_offer_volume += v
          stats.added_best_offer_count += 1
                   
        #base_price = best_offer_price
        #cumulative_volume = sum([order.size for order in ob.offers if order.price <= p])
      else: 
        stats.bid_tr8dr += (p - best_bid_price) / best_bid_price 
        stats.added_bid_volume += v
        stats.added_bid_count += 1
        if p >= best_bid_price:
          stats.added_best_bid_volume += v
          stats.added_best_bid_count += 1
        #base_price = best_bid_price
        #cumulative_volume = sum([order.size for order in ob.bids if order.price >= p])
    elif action_type == DELETE_ACTION_TYPE:
      if side == OFFER_SIDE:
        stats.offer_tr8dr -= (p - best_offer_price) / best_offer_price 
        if p <= best_offer_price: 
          stats.filled_offer_volume += v 
          stats.filled_offer_count += 1 
        else:
          stats.canceled_offer_volume += v
          stats.canceled_offer_count += 1
      else:
        stats.bid_tr8dr -= (p - best_bid_price) / best_bid_price 
        if p >= best_bid_price:
          stats.filled_bid_volume += v
          stats.filled_bid_count += 1
        else:
          stats.canceled_bid_volume += v
          stats.canceled_bid_count += 1
  stats.deleted_bid_volume = stats.canceled_bid_volume + stats.filled_bid_volume
  stats.deleted_bid_count = stats.canceled_bid_count + stats.filled_bid_count
  stats.deleted_offer_volume = stats.canceled_offer_volume + stats.filled_offer_volume 
  stats.deleted_offer_count = stats.canceled_offer_count + stats.filled_offer_count
  return stats