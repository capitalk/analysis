
from optparse import OptionParser
import os, os.path, datetime
import progressbar  
import cloud
from extractor import extractor
import hdf 
import cloud_helpers

def process_local_file(input_filename, dest_1ms, dest_100ms, max_books = None, heap_profile=False):
  print "Start time:", datetime.datetime.now()
  
  frames_1ms, frames_100ms, header = \
    extractor.run(input_filename,  max_books = max_books, 
      heap_profile = heap_profile)

  if dest_1ms:
    hdf.dict_to_hdf(frames_1ms, dest_1ms, header, extractor.feature_names() )
  if dest_100ms:
    hdf.dict_to_hdf(frames_100ms, dest_100ms, header, extractor.feature_names())
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
      
      feature_names = extractor.feature_names()
      if not overwrite and \
         hdf.complete_hdf_exists(dest_filename_1ms, feature_names) and \
         hdf.complete_hdf_exists(dest_filename_100ms, feature_names):
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
  
  input_filename = cloud_helpers.download_file_from_s3(input_bucket_name, input_key_name)   
  dest_1ms, dest_100ms = output_filenames(input_filename, cloud_helpers.scratch_dir())
  
  out_key_name_1ms = os.path.split(dest_1ms)[1]  
  out_key_name_100ms = os.path.split(dest_100ms)[1]
  
  if output_bucket_name_1ms is None:
     output_bucket_name_1ms = input_bucket_name + "-hdf-1ms"
  
  if output_bucket_name_100ms is None:
     output_bucket_name_100ms = input_bucket_name + "-hdf"
     
  feature_names = extractor.feature_names()
  
  if not overwrite and \
     cloud_helpers.hdf_already_on_s3(output_bucket_name_1ms, out_key_name_1ms, feature_names) and \
     cloud_helpers.hdf_already_on_s3(output_bucket_name_100ms, out_key_name_100ms, feature_names):
  
    print "HDFs on S3 have same features, so skipping this input..."
    return
  else:
    print "HDFs either not on S3 or have different features..."
  
  # In some weird situation we might have a local copy of the HDF already 
  # finished but it just might not have been uploaded yet       
  if not overwrite and hdf.complete_hdf_exists(dest_1ms, feature_names) and \
     hdf.complete_hdf_exists(dest_100ms, feature_names):
    print "Found finished HDFs on local storage..."
    header = hdf.header_from_hdf_filename(dest_1ms) 
  else:
    print "Running feature generator..."
    header = process_local_file(input_filename, dest_1ms, dest_100ms)

  print "Header:", header
  print "Uploading 1ms feature file", dest_1ms, "to", output_bucket_name_1ms, "/", out_key_name_1ms
  cloud_helpers.upload_file_to_s3(\
    dest_1ms, output_bucket_name_1ms, out_key_name_1ms, header)
  print "Uploading 100ms feature file", dest_100ms, "to", output_bucket_name_100ms, "/", out_key_name_100ms
  cloud_helpers.upload_file_to_s3(\
    dest_100ms, output_bucket_name_100ms, out_key_name_100ms, header)
  
  
def process_s3_files(input_bucket_name, key_glob = '*', 
      output_bucket_name_1ms = None, 
      output_bucket_name_100ms = None, 
      overwrite = False, 
      use_cloud = True):
      
  if output_bucket_name_1ms is None:
    output_bucket_name_1ms = input_bucket_name + "-hdf-1ms"
  
  if output_bucket_name_100ms is None:
    output_bucket_name_100ms = input_bucket_name + "-hdf" 
  
  matching_keys = cloud_helpers.get_matching_key_names(input_bucket_name, key_glob)
  
  s3_cxn = cloud_helpers.get_s3_cxn()    
  # create output buckets if they don't already exist
  # it's better to do this before launching remote computations 
  s3_cxn.create_bucket(output_bucket_name_1ms)
  s3_cxn.create_bucket(output_bucket_name_100ms)
  
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