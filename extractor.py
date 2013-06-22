import features 
from feature_pipeline import FeaturePipeline


extractor = FeaturePipeline()
extractor.add_feature('t', features.millisecond_timestamp)
extractor.add_feature('bid', features.best_bid)
extractor.add_feature('offer', features.best_offer)
extractor.add_feature('bid_range', features.bid_range)
extractor.add_feature('offer_range', features.offer_range)

extractor.add_feature('spread', features.spread)

extractor.add_feature('locked', features.locked)
extractor.add_feature('crossed', features.crossed)

extractor.add_feature('midprice', features.midprice)
extractor.add_feature('bid_vwap', features.bid_vwap)
extractor.add_feature('offer_vwap', features.offer_vwap)

extractor.add_feature('bid_slope', features.bid_slope)
extractor.add_feature('offer_slope', features.offer_slope)

extractor.add_feature('offer_vol', features.best_offer_volume)
extractor.add_feature('bid_vol', features.best_bid_volume)

extractor.add_feature('total_bid_vol', features.bid_volume)
extractor.add_feature('total_offer_vol', features.offer_volume)
extractor.add_feature('t_mod_1000', features.fraction_of_second)
extractor.add_feature('message_count', features.message_count, sum_100ms=True)
# V3 orderbook action  features
extractor.add_feature('bid_tr8dr', features.bid_tr8dr)
extractor.add_feature('offer_tr8dr', features.offer_tr8dr)
extractor.add_feature('tr8dr', features.tr8dr)

extractor.add_feature('added_total_bid_vol', features.added_bid_volume, sum_100ms=True)
extractor.add_feature('added_total_bid_count', features.added_bid_count, sum_100ms=True)
extractor.add_feature('added_total_offer_vol', features.added_offer_volume, sum_100ms=True)
extractor.add_feature('added_total_offer_count', features.added_offer_count, sum_100ms=True)

extractor.add_feature('added_best_bid_vol', features.added_best_bid_volume, sum_100ms=True)
extractor.add_feature('added_best_bid_count', features.added_best_bid_count, sum_100ms=True)
extractor.add_feature('added_best_offer_vol', features.added_best_offer_volume, sum_100ms=True)
extractor.add_feature('added_best_offer_count', features.added_best_offer_count, sum_100ms=True)

extractor.add_feature('deleted_total_bid_vol', features.deleted_bid_volume, sum_100ms=True)
extractor.add_feature('deleted_total_bid_count', features.deleted_bid_count, sum_100ms=True)
extractor.add_feature('deleted_total_offer_vol', features.deleted_offer_volume, sum_100ms=True)
extractor.add_feature('deleted_total_offer_count', features.deleted_offer_count, sum_100ms=True)

extractor.add_feature('filled_bid_vol', features.filled_bid_volume, sum_100ms=True)
extractor.add_feature('filled_bid_count', features.filled_bid_count, sum_100ms=True)
extractor.add_feature('filled_offer_vol', features.filled_offer_volume, sum_100ms=True)
extractor.add_feature('filled_offer_count', features.filled_offer_count, sum_100ms=True)

extractor.add_feature('canceled_bid_vol', features.canceled_bid_volume, sum_100ms=True)
extractor.add_feature('canceled_bid_count', features.canceled_bid_count, sum_100ms=True)
extractor.add_feature('canceled_offer_vol', features.canceled_offer_volume, sum_100ms=True)
extractor.add_feature('canceled_offer_count', features.canceled_offer_count, sum_100ms=True)
