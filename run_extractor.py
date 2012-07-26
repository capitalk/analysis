
from optparse import OptionParser
import os, os.path
import h5py, datetime
import boto
import fnmatch 
import progressbar  
import cloud
from extractor import extractor
from hdf import header_from_hdf_filename, dict_to_hdf

import sys

sys.path.append('..')
from cloud_helpers
def same_features(f1, f2):
  s1 = set(f1)
  s2 = set(f2)
  same = s1 == s2
  if not same:
    print "Different features:", \
      s1.symmetric_difference(s2)
  return same

 # file exists and 'finished' flag is true
def file_already_done(filename):
  if not os.path.exists(filename): 
    print "Doesn't exist"
    return False
  try:
      f = h5py.File(filename, 'r')
      attrs = f.attrs
      finished = 'finished' in attrs and attrs['finished']
      has_ccy = 'ccy1' in attrs and 'ccy2' in attrs 
      has_date = 'year' in attrs and 'month' in attrs and 'day' in f.attrs 
      has_venue = 'venue' in attrs 
      has_features = 'features' in attrs
      if has_features:
        have_same_features = same_features(set(attrs['features']), extractor.feature_names())
      else:
        have_same_features = False    
      f.close()
      return finished and has_ccy and has_date and has_venue and \
        has_features and have_same_features  
  except:
      import sys
      print sys.exc_info()
      return False

def process_local_file(input_filename, dest_1ms, dest_100ms, max_books = None, heap_profile=False):
  print "Start time:", datetime.datetime.now()
  
  frames_1ms, frames_100ms, header = \
    extractor.run(input_filename,  max_books = max_books, 
      heap_profile = heap_profile)

  if dest_1ms:
    dict_to_hdf(frames_1ms, dest_1ms, header, extractor.feature_names() )
  if dest_100ms:
    dict_to_hdf(frames_100ms, dest_100ms, header, extractor.feature_names() )
  return header 
    
    
def output_filenames(input_path, feature_dir = None):
  assert not os.path.isdir(input_path)
  base_dir, input_name  = os.path.split(input_path)
  
  if feature_dir is None: 
    feature_dir = os.path.join(base_dir, "features")
  
  if not os.path.exists(feature_dir): 
    os.makedirs(feature_dir)
  
  no_ext = os.path.basename(input_name).split('.')[0]
  dest_1ms = os.path.join(feature_dir, no_ext + "_1ms.hdf")
  dest_100ms = os.path.join(feature_dir, no_ext + "_100ms.hdf")
  return dest_1ms, dest_100ms

def process_local_dir(input_path, output_dir = None, max_books = None,
  heap_profile=False, overwrite = False):
  
  if not os.path.exists(input_path):
    print "Specified path does not exist: ", input_path
    exit()
        
  if os.path.isdir(input_path):
    files = os.listdir(input_path)
    basedir = input_path
  else:
    basedir = os.path.split(input_path)[0]
    files = [os.path.basename(input_path)]
  
  count = 0
  for filename in files:
    if filename.endswith('.csv') or filename.endswith('.csv.gz'):
      count += 1
      input_filename = os.path.join(basedir, filename)
      print "----"
      print "Processing  #", count, " : ", input_filename
      
      dest_filename_1ms, dest_filename_100ms = \
        output_filenames (input_filename, output_dir)
      
      if not overwrite and \
         file_already_done(dest_filename_1ms) and \
         file_already_done(dest_filename_100ms):
          print "Skipping %s found data files %s" \
            % (input_filename, [dest_filename_1ms, dest_filename_100ms])
      else:
          process_local_file(input_filename,
             dest_filename_1ms, dest_filename_100ms, 
             max_books, heap_profile)
    else:
      print "Unknown suffix for", filename  



def process_s3_file(input_bucket_name, input_key_name, 
    output_bucket_name_1ms = None, 
    output_bucket_name_100ms = None, 
    overwrite = False):
  
  if output_bucket_name_1ms is None:
     output_bucket_name_1ms = input_bucket_name + "-hdf-1ms"
  
  if output_bucket_name_100ms is None:
     output_bucket_name_100ms = input_bucket_name + "-hdf"
     
  if os.access('/scratch/sgeadmin', os.F_OK | os.R_OK | os.W_OK):
    print 'Using /scratch/sgeadmin for local storage'
    tempdir = '/scratch/sgeadmin'
  elif os.access('/tmp', os.F_OK | os.R_OK | os.W_OK):
    print 'Using /tmp for local storage'
    tempdir = '/tmp'
  else:
    print 'Using ./ for local storage'
    tempdir = './'
  input_filename = os.path.join(tempdir, input_key_name)
  
  s3_cxn = get_s3_cxn() 
  in_bucket = s3_cxn.get_bucket(input_bucket_name)
  assert in_bucket is not None
  in_key = in_bucket.get_key(input_key_name)
  if in_key is None:
    raise RuntimeError(\
      "Key Not Found: bucket = " + input_bucket_name  \
      + ", input_key = " + input_key_name)
  print "Downloading", input_key_name, "to", input_filename, "..."
  if os.path.exists(input_filename) and \
     os.path.getsize(input_filename) == in_key.size:
    print "Already downloaded", input_filename, "from S3"
  else:
    get_s3_key_contents()
      
  dest_1ms, dest_100ms = output_filenames(input_filename, tempdir)
  
  filename_1ms = os.path.split(dest_1ms)[1]  
  filename_100ms = os.path.split(dest_100ms)[1] 
  
  out_bucket_1ms = s3_cxn.get_bucket(output_bucket_name_1ms)  
  out_bucket_100ms = s3_cxn.get_bucket(output_bucket_name_100ms)
    
  out_key_1ms = out_bucket_1ms.get_key(filename_1ms)
  out_key_100ms = out_bucket_100ms.get_key(filename_100ms)
  
  feature_set = set(extractor.feature_names())
  
  if not overwrite and out_key_1ms is not None and out_key_100ms is not None:
    print "Found", out_key_1ms, "and", out_key_100ms, "already on S3"
    features_1ms = out_key_1ms.get_metadata('features')
    features_100ms = out_key_100ms.get_metadata('features')
    if features_1ms is not None and features_100ms is not None and \
      same_features(feature_set, features_1ms) and \
      same_features(feature_set, features_100ms):
      print "HDFs on S3 have same features, so skipping this input..."
      return
    else:
      print "HDFs on S3 have different features, so regenerating them..."
        
  if not overwrite and file_already_done(dest_1ms) and file_already_done(dest_100ms):
    print "Found finished HDFs on local storage..."
    header = header_from_hdf_filename(dest_1ms) 
  else:
    print "Running feature generator..."
    header = process_local_file(input_filename, dest_1ms, dest_100ms)

  if out_key_1ms is None:
    out_key_1ms = boto.s3.key.Key(out_bucket_1ms)
    out_key_1ms.key = filename_1ms
    
  if out_key_100ms is None:
    out_key_100ms = boto.s3.key.Key(out_bucket_100ms)
    out_key_100ms.key = filename_100ms
 
  print "Uploading 1ms feature file..."
  set_s3_key_contents(out_key_1ms, dest_1ms, header)
 
  print "Uploading 100ms feature file..."
  # retry, since boto can time out for bizarre reasons
  set_s3_key_contents(out_key_100ms, dest_100ms, header)
  
def process_s3_files(input_bucket_name, key_glob = '*', 
      output_bucket_name_1ms = None, 
      output_bucket_name_100ms = None, 
      overwrite = False, 
      use_cloud = True):
      
  if output_bucket_name_1ms is None:
    output_bucket_name_1ms = input_bucket_name + "-hdf-1ms"
  
  if output_bucket_name_100ms is None:
    output_bucket_name_100ms = input_bucket_name + "-hdf" 
  
  s3_cxn = get_s3_cxn()    
  in_bucket = s3_cxn.get_bucket(input_bucket_name)
  
  # create output buckets if they don't already exist
  # it's better to do this before launching remote computations 
  s3_cxn.create_bucket(output_bucket_name_1ms)
  s3_cxn.create_bucket(output_bucket_name_100ms)
  
  matching_keys = []
  for k in in_bucket:
    if fnmatch.fnmatch(k.name, key_glob):
      matching_keys.append(k.name)
     
  if use_cloud:
    print "Launching %d jobs" % len(matching_keys)
    def do_work(key_name):
      return process_s3_file(
        input_bucket_name, 
        key_name, 
        output_bucket_name_1ms, 
        output_bucket_name_100ms, 
        overwrite)
    label = "Generate HDF files for %s/%s" % (input_bucket_name, key_glob)
    jids = cloud.map(do_work, matching_keys, _type = 'f2', _label=label )
    
    progress = progressbar.ProgressBar(len(jids)).start()
    n_finished = 0
    for _ in cloud.iresult(jids):
      n_finished += 1
      progress.update(n_finished)
    progress.finish()
  else:
    print "Running locally..."
    print "%d keys match the pattern \'%s\'" % (len(matching_keys), key_glob)
    for key in matching_keys:
      process_s3_file(input_bucket_name, key, output_bucket_name_1ms, output_bucket_name_100ms)
  print "Done!"
  
parser = OptionParser(usage = "usage: %prog [options] path")
parser.add_option("-m", "--max-books", dest="max_books",
  type="int", help="maximum number of order books to read", default=None)
  
parser.add_option("-o", "--overwrite", dest="overwrite", default=False, 
  action="store_true", help="Overwrite existing HDF files")
  
parser.add_option("-d", "--feature-dir",
  dest="feature_dir", default=None, type="string",
  help="which directory should we write feature files to")

parser.add_option('--heap-profile', default=False, action="store_true",
  help='Show heap statistics after exec (local only)')

if __name__ == '__main__':
  (options, args) = parser.parse_args()
  print "Args = ", args
  print "Options = ", options
  if len(args) != 1:
    parser.print_help()
  elif args[0].startswith('s3://'):
    bucket, _, pattern = args[0].split('s3://')[1].partition('/')
    assert bucket and len(bucket) > 0
    assert pattern and len(pattern) > 0
    print "Bucket = %s, pattern = %s" % (bucket, pattern)
    process_s3_files(bucket, pattern, overwrite = options.overwrite, use_cloud=True)
  else:
    process_local_dir(args[0],
      options.feature_dir,
      options.max_books, 
      options.heap_profile, 
      overwrite = options.overwrite)