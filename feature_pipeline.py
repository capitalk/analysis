import bisect 
import math
import numpy as np

import progressbar
import order_book_parser 

from order_book import compute_stats

def show_heap_info():
  from guppy import hpy
  heap = hpy().heap()
  print heap
  print heap[0].rp
  print heap[0].byid

def time_weighted_average_100ms(feature_1ms, milliseconds, 
      start_indices, end_indices, frame_end_times):
    """
    Given a series of unevenly spaced 1ms feature samples, 
    average every 100ms frame by weighing each 1ms value by what percent
    of the frame it survives. 
    - feature_1ms : data we're trying to aggregate
    - milliseconds : at which time did each feature sample arrive?  
    - start_indices : which index in the uneven data corresponds to the start
        of a 100ms frame
    - end_indices : which index in the uneven is one 
        past the end of a 100ms frame
    - frame_end_times : 1ms past time each frame ends 
    """
    
    assert len(start_indices) == len(end_indices)
    assert len(frame_end_times) == len(start_indices)
    assert len(milliseconds) == len(feature_1ms)
    
    feature_100ms = np.zeros(len(frame_end_times))
   
    tick_counts = end_indices - start_indices 
    
    # either last sample in the 100ms frame, or, 
    # if the frame was empty, the last received sample before
    last_sample = feature_1ms[end_indices-1]
    
    no_messages_idx = np.where(tick_counts == 0)[0]
    if len(no_messages_idx) > 0: 
      feature_100ms[no_messages_idx] = last_sample.take(no_messages_idx)
    
    one_message_idx = np.where(tick_counts == 1)[0]
    if len(one_message_idx) > 0:
      one_message_1ms_idx = start_indices.take(one_message_idx)
      if 0 in one_message_idx:
        feature_100ms[0] = feature_1ms[0]
        one_message_idx = one_message_idx[1:]
        one_message_1ms_idx = one_message_1ms_idx[1:]
        
      one_message_frame_end_times = frame_end_times.take(one_message_idx)
      one_message_frame_start_times = one_message_frame_end_times - 100
      one_message_time_since_frame_start = \
        milliseconds[one_message_1ms_idx] - one_message_frame_start_times
  
      past = feature_1ms.take(one_message_1ms_idx-1) * one_message_time_since_frame_start
      curr = feature_1ms.take(one_message_1ms_idx) * (100 - one_message_time_since_frame_start)
      feature_100ms[one_message_idx] = (past + curr) / 100.0
    
    multiple_message_idx = np.where(tick_counts > 1)[0]
    # keep around big arrays to prevent from having to keep recomputing this
    weights = np.zeros(101, dtype='int')
    
    if 0 in multiple_message_idx:
      n_ticks = tick_counts[0]
      frame_end_t = frame_end_times[0]
      frame_start_t = frame_end_t - 100 
      start_idx = 0
      end_idx = end_indices[0] 
      # weights are time between frame arrivals
      relative_t = milliseconds[:end_idx] - frame_start_t
      weights[:n_ticks-1] = np.diff(relative_t)
      weights[n_ticks-1] = 100 - relative_t[-1]
      weight_slice = weights[:n_ticks]
      feature_slice = feature_1ms[:end_idx]  
      total_weight = np.sum(weight_slice)  
      feature_100ms[0] = np.dot(weight_slice, feature_slice) / total_weight
      
      # drop  0 from the list of indices
      multiple_message_idx = multiple_message_idx[1:]
    if len(multiple_message_idx) > 0:  
      for i in multiple_message_idx: 
        n_ticks = tick_counts[i]
        frame_end_t = frame_end_times[i]
        frame_start_t = frame_end_t - 100 
        start_idx = start_indices[i] 
        end_idx = end_indices[i] 
        # reach back to the last 1ms in the previous frame
        relative_t = milliseconds[start_idx:end_idx] - frame_start_t
        weights[0] = relative_t[0]
        weights[1:n_ticks] = np.diff(relative_t)
        weights[n_ticks] = 100 - relative_t[-1]
        weight_slice = weights[:n_ticks+1]
              
        feature_slice = feature_1ms[start_idx-1:end_idx] 
        feature_100ms[i] = np.dot(weight_slice, feature_slice) / 100.0 
    return feature_100ms 

def sum_100ms(feature_1ms, start_indices, end_indices):
    n = len(start_indices)
    feature_100ms = np.zeros(n, dtype='float')
    diffs = end_indices - start_indices 
    # if there's only one 1ms sample in the whole 100ms period, 
    # then just copy over that sample 
    only_one = (diffs==1)
    feature_100ms[only_one] = feature_1ms[np.array(start_indices[only_one])]
    # now loop over all the periods when more than one sample needs to be 
    # averaged 
    for i in np.where(diffs>1)[0]: 
      ticks = feature_1ms[start_indices[i] :end_indices[i]]
      feature_100ms[i] = np.sum(ticks)
    return feature_100ms

    

# given a list of books, return a dictionary of feature vectors 
def features_from_books(books, feature_fns, show_progress=False, output=True):
    # these should all be from the same day, so discard any with days
    # other than a book in the middle 
    valid_day = books[len(books)/2].day
    
    # scan for orderbook with non-empty bids and offers
    validBooks = [ book for book in books if \
      book.bids and book.offers and book.day == valid_day]
    # generator expression to count all invalid books 
    
    numInvalid = len(books) - len(validBooks)
    
    if output:
      print "Keeping %d of %d order books (%d dropped)" % \
        (len(validBooks), len(books), numInvalid)
            
    n = len(validBooks)
    nfeatures = len(feature_fns)
    
    if output: 
      print "Extracting %d features...." % nfeatures

    result = {}
    for feature_name in feature_fns.keys(): 
      dtype = 'int' if feature_name == 't' else 'float'
      result[feature_name] = np.zeros(n, dtype=dtype)
    
    if show_progress: progress = progressbar.ProgressBar(n).start()  
    for (i,book) in enumerate(validBooks):
      stats = compute_stats(book)
      for (feature_name, fn) in feature_fns.iteritems():
        if fn:
          result[feature_name][i] = fn(book, stats) 
        else:
          result[feature_name][i] = stats[feature_name]
      if show_progress: progress.update(i)
    progress.finish()   
          
    if output: print 
    return result   

        
def features_from_filename(
  filename, 
  feature_fns, 
  debug=False, 
  max_books=None, 
  show_progress=False, 
  output=True, 
  heap_profile = False):
    if heap_profile: 
      print "=== Heap before parsing orderbooks ==="
      show_heap_info()
    header, books = order_book_parser.read_books_from_filename(filename, debug, end=max_books)
    if heap_profile: 
      print "=== Heap after parsing orderbooks ==="
      show_heap_info()
    if max_books: 
      books = books[:max_books]
    features = \
      features_from_books(books, feature_fns, 
        show_progress=show_progress, output=output)
    del books
    if heap_profile: 
      print "=== Heap after extracting features ==="
      show_heap_info()
      
    return header, features 



def make_1ms_group_indices(times, unique_times = None):
  if unique_times is None:
    unique_times = np.unique(times)
    
  num_unique_times = len(unique_times)
  num_raw_times = len(times)
  group_start_indices = np.zeros(num_unique_times, dtype='int')
  search_start = 0 
  for i, t in enumerate(unique_times):
    first_index = bisect.bisect_left(times, t, search_start, num_raw_times)
    group_start_indices[i] = first_index 
    search_start = first_index 
  group_end_indices = np.concatenate( [group_start_indices[1:], [num_raw_times]])
  return group_start_indices, group_end_indices  


def aggregate_1ms_frames(features, frame_reducers, output=True): 
  times = features['t']
  unique_times = np.unique(times)
  
  unique_times.sort()
  num_unique_times = len(unique_times)
  if output:
    print "Found",  len(times), "timestamps, of which", num_unique_times, "are unique"
  
  if output: 
    print "Computing 1ms frame indices..." 
  
  window_starts, window_ends = make_1ms_group_indices(times, unique_times)
  t_diff = np.concatenate([[0], np.diff(unique_times)])
  frames_1ms = {'t': unique_times,  'time_since_last_message': t_diff}
  
  if output: 
    print "Aggregating 1ms frames..." 
  
  nreducers = len(frame_reducers)
  if output: progress = progressbar.ProgressBar(nreducers).start()
  counter = 0 
  for name, fn in frame_reducers.items():
      if name != 't':
          raw = features[name] 
          result = np.zeros(num_unique_times)
          lengths = window_ends - window_starts
          only_one_idx = (lengths==1)
          result[only_one_idx] = raw[window_starts[only_one_idx]]
          
          for i in np.where(lengths > 1)[0]: 
            curr_slice = raw[window_starts[i]:window_ends[i]] 
            result[i] = fn(curr_slice)
            
          frames_1ms[name] = result
      counter += 1
      if output: progress.update(counter)
  if output: progress.finish() 
  if output: print 
  return frames_1ms 
    
def aggregate_100ms_frames(frames_1ms, sum_100ms_dict, output=True): 
  
    
    
    # We're starting with a sparse set of samples coming in at sporadic
    # 1ms timestamps and trying to aggregating them into a uniform sequence
    # of 100ms frames. 
    # Each 100ms timestamp is responsible for aggregating every 1ms tick 
    # which occured from t-99 until (inclusive)
    
    milliseconds = frames_1ms['t']
    start_millisecond = milliseconds[0]
    end_millisecond = milliseconds[-1]
     
    # we start by finding by rounding the first and last 1ms timestamps
    # to their nearest 100ms time 
    round_start = int(math.ceil(start_millisecond / 100.0) * 100)
    round_end = int(math.ceil(end_millisecond / 100.0) * 100)

    if output: 
      print "Generating 100ms frame indices..."
    
    # frames start on the 0th millisecond into a frame and end at the 99th
    # millisecond
    # for example, if we had ticks at A:0, B:99, C:100, D:101 
    # Then {A,B} becomes frame 100ms, {C,D} become frame 200ms
    
    # if first tick is at 0 milliseconds, don't want to create a 0ms frame
    if milliseconds[0] == round_start:
      frame_end_times = np.arange(round_start + 100, round_end+1, 100)
    else:
      frame_end_times = np.arange(round_start, round_end+1, 100)  

    n_100ms_frames = len(frame_end_times)
    # map each 100ms frame to its indices in vectors of 1ms data 
    frame_start_indices = np.zeros(n_100ms_frames, dtype = 'int')
    frame_end_indices = np.zeros(n_100ms_frames, dtype = 'int')
    empty_frames = np.zeros(n_100ms_frames, dtype='bool')

    for (i, frame_t) in enumerate(frame_end_times):
      
        # a frame from 0ms - 100ms includes the left boundary but 
        # excludes the right one 
        frame_end_t = frame_t - 1
        frame_start_t = frame_t - 100
        # search for leftmost occurrence of frame start time 
        start_idx = bisect.bisect_left(milliseconds, frame_start_t)
        if milliseconds[start_idx] > frame_end_t:
            empty_frames[i] = True 
            end_idx = start_idx 
        else:
            # search for leftmost occurrence of frame end time 
            end_idx = bisect.bisect_right(milliseconds, frame_end_t, start_idx) 

        # start indices exclude time (t - 100)
        frame_start_indices[i] = start_idx
        # end indices include time t
        frame_end_indices[i] = end_idx

    features_100ms = {'t': frame_end_times, 'null_100ms_frame': empty_frames}

    if output:
      print "Aggregating 100ms frames..." 

    n_completed = 0 
    if output: 
      progress = progressbar.ProgressBar(len(frames_1ms)).start()
    for fName, vec_1ms in frames_1ms.items(): 
        #if output: print "  ", fName 

        # time, and counts get compute separately from normal features 
        if fName != 't' and fName != 'time_since_last_message': 
            if sum_100ms_dict[fName]:
                features_100ms[fName] = \
                  sum_100ms(vec_1ms, frame_start_indices, frame_end_indices)
            else:
                features_100ms[fName] = \
                  time_weighted_average_100ms(
                    vec_1ms,
                    milliseconds,  
                    frame_start_indices, 
                    frame_end_indices, 
                    frame_end_times)
        n_completed += 1
        if output:
          progress.update(n_completed)
    if output:
      progress.finish()
    print "Computing time between messages and 1ms frame counts..."         
    # compute the time since the last message for all frames, even null ones 
    # this is done by looking at the millisecond timestamp of the message before
    # the frame's end. for null frames this will reach back to a time > 100ms.
    # Also, compute the total message count over the frame, which 
    # is simply summed over the individual 1ms counts 
    time_since_last_message = np.zeros(n_100ms_frames, dtype='int')
    small_frame_count = np.zeros(n_100ms_frames, dtype='int')

    for i in xrange(n_100ms_frames):
        time_since_last_message[i] = \
          frame_end_times[i] - milliseconds[frame_end_indices[i] - 1]
        small_frame_count[i] = frame_end_indices[i]  - frame_start_indices[i] + 1 
    features_100ms['time_since_last_message'] = time_since_last_message 
    features_100ms['1ms_frame_count'] = small_frame_count 

    return features_100ms
    
    
class FeaturePipeline:
    
    def __init__(self):
        self.feature_fns = {} 
        #maps feature name to single (name,fn) pair
        self.frame_reducers_1ms = {}
        self.sum_100ms_feature = {} 
            

    def add_feature(self, name, fn, frame_reducer_1ms=lambda x: x[-1], sum_100ms = False):
        self.feature_fns[name] = fn
        self.frame_reducers_1ms[name] = frame_reducer_1ms
        self.sum_100ms_feature[name] = sum_100ms 
    
    
    def feature_names(self):
      return self.feature_fns.keys()
    
    def feature_name_set(self):
      return set(self.feature_names())
            
    def run(self, input_filename, max_books = None, heap_profile = False):
        
        header, raw_features = features_from_filename(input_filename,  
          self.feature_fns, 
          max_books=max_books, 
          show_progress=True, 
          heap_profile = heap_profile)
          
        assert 'ccy' in header and len(header['ccy']) == 2
        header['features'] = self.feature_fns.keys()
        
        frames_1ms = aggregate_1ms_frames(raw_features, self.frame_reducers_1ms)
        del raw_features 
        frames_100ms = aggregate_100ms_frames(frames_1ms, self.sum_100ms_feature)
        
        return frames_1ms, frames_100ms, header 
       
       
        
