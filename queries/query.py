# ugh, why doesn't python allow relative imports outside packages?
# or allow parts of a package to run standalone?
import sys
sys.path.append("..")

import hdf
import cloud_helpers
import cloud 
import progressbar 



def wrap_hdf_mapper(fn):
  def do_work(bucket, key_name):
    hdf_file = cloud_helpers.download_hdf_from_s3(bucket, key_name)
    result = fn(hdf_file)
    hdf_file.close()
    return result  
  return do_work 

def wrap_dataframe_mapper(fn):
  def do_work(bucket, key_name):
    hdf_file = cloud_helpers.download_hdf_from_s3(bucket, key_name)
    dataframe = hdf.dataframe_from_hdf(hdf_file)
    result = fn(dataframe)
    hdf_file.close()
    return result
  return do_work
  
def run(bucket, key_pattern = None, 
  map_hdf = None, map_dataframe = None,
  init = None, combine = None,  post_process = None, 
  accept_none_as_result = False, label = None, _type = 'f2'):
  
  """
  Runs query functions over collections of HDF files on S3. 
  You *must* provide: 
    - either the bucket and key_pattern (ie: "capk-fxcm" and "*2012*")
      or, a concatenation of the two separated by a slash ("capk-fxcm/*2012*")
    - either map_hdf (a function which takes HDF objects and returns a result)
      or map_dataframe (which takes pandas.DataFrames and returns a result)
  If you are running a query which returns a result you must provide:
    - init : the initial value of the computation
    - combine : a function which takes the value thus-far accumulated and 
      the result of a single mapper and combines them into a new accumulated
      result
  
  Additional arguments:
     - accept_none_as_result : Don't warn me if my mappers are returning None
     - label : label shown on picloud's job manager
     - _type : which picloud instance type to use (one of 'f2', 'm1', 'c1', 'c2')
  """
     
  if key_pattern is None:
    bucket, key_pattern = cloud_helpers.parse_bucket_and_pattern(bucket)
  if len(key_pattern) == 0:
    key_pattern = '*'
  key_names = cloud_helpers.get_matching_key_names(bucket, key_pattern)

  if combine: 
    assert init is not None

  if init and hasattr(init, '__call__'):
    acc = init()
  else:
    # if we can't call init assume it's either None or some value
    acc = init
  
  if map_hdf is not None:
    assert map_dataframe is None
    do_work = wrap_hdf_mapper(map_hdf)
  else:
    assert map_dataframe is not None
    assert map_hdf is None
    do_work = wrap_dataframe_mapper(map_dataframe)
  
  if label is None:
    label = "Mapping %s over %s/%s" % \
      ( (map_hdf if map_hdf else map_dataframe), bucket, key_pattern)
      
  jids = cloud.map(lambda name: do_work(bucket, name), \
       key_names, _type = _type, _label= label)
  try:
    progress = progressbar.ProgressBar(len(jids)).start()
    for (i, result) in enumerate(cloud.iresult(jids)):
      if result is None and not accept_none_as_result:
        print "Job #", jids[i], key_names[i], "returned None"
      elif combine:
        # client-side reduction! Be careful about not doing too much
        # work here
        new_acc = combine(acc, result)
        if new_acc is None and acc is not None:
          print "Warning: You it looks like your combine fn unintentionally returns None"
        acc = new_acc
      progress.update(i+1)
  except KeyboardInterrupt:
    print "Keyboard interrupt received, killing active workers..."
    cloud.kill(jids)
  except:
    cloud.kill(jids)
    raise
  finally:
    progress.finish()
    
  if post_process:
    accumulator = post_process(accumulator)
  return accumulator


