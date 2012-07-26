
import boto 
from ssl import SSLError  
import socket
import time 
import cloud
import fnmatch
import os
import random
import h5py

PICLOUD_ID = 2579
PICLOUD_SECRET_KEY = 'f228c0325cf687779264a0b0698b0cfe40148d65'
cloud.setkey(PICLOUD_ID, PICLOUD_SECRET_KEY)

AWS_ACCESS_KEY = 'AKIAJSCF3K3HKREPYE6Q'
AWS_SECRET_KEY = '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'

def get_s3_cxn():
  # wait two minutes before retrying S3 operations
  socket.setdefaulttimeout(120)
  s3_cxn = boto.connect_s3(AWS_ACCESS_KEY, AWS_SECRET_KEY)
  if s3_cxn is None:
    raise RuntimeError("Couldn't connect to S3")
  else:
    return s3_cxn

def scratch_dir():
  # if running on starcluster machine, try to use ephemeral storage
  if os.access('/scratch/sgeadmin', os.F_OK | os.R_OK | os.W_OK):
    return '/scratch/sgeadmin'
  elif os.access('/tmp', os.F_OK | os.R_OK | os.W_OK):
    return '/tmp'
  else:
    return './'


def get_bucket(bucket_name, s3_cxn = None):
  if not s3_cxn:
    s3_cxn = get_s3_cxn()
  try:  
     b = s3_cxn.get_bucket(bucket_name)
  # retry if communication times out 
  except SSLError:
    time.sleep(1)
    b = s3_cxn.get_bucket(bucket_name)

  if b is not None:
    return b
  else:
    raise RuntimeError("Could not find S3 bucket " + bucket_name)

def get_key(bucket_name, key_name, s3_cxn = None):
  b = get_bucket(bucket_name, s3_cxn = s3_cxn)
  try:
    k = b.get_key(key_name)
  except SSLError:
    time.sleep(1)
    k = b.get_key(key_name)

  if k is not None:
    return k
  else:
    raise RuntimeError(\
      "Could not find key " + key_name + " in bucket " + bucket_name)

def parse_bucket_and_pattern(s):
  s = s.strip()
  if s.startswith("s3://"):
    s = s[5:]
  if s[-1] == '/':
    s = s[:-1]
  bucket_name, _, key_pattern = s.partition('/')
  assert len(bucket_name) > 0
  if len(key_pattern) == 0:
    key_pattern = '*'
  return bucket_name, key_pattern

# key_pattern isn't given then assume it's part of bucket_name 
def get_matching_keys(bucket_name, key_pattern, s3_cxn = None, output=True):
  
  if output:
    print "Bucket = %s, pattern = %s" % (bucket_name, key_pattern)
  assert bucket_name and len(bucket_name) > 0
  assert key_pattern and len(key_pattern) > 0
  
  if not any([wildcard in key_pattern for wildcard in ['*', '?', '[', ']', '!']]):
    # if there no special characters in the key pattern treat it as just an 
    # ordinary name 
    matching_keys = [get_key(bucket_name, key_pattern, s3_cxn = s3_cxn)]
  else:
    b = get_bucket(bucket_name, s3_cxn = s3_cxn)
    matching_keys = []
    for k in b:
      if fnmatch.fnmatch(k.name, key_pattern):
        matching_keys.append(k)
  if output:
    print "Found", len(matching_keys), "matching keys"
  return matching_keys

# if key_pattern not given, assume that it's part of bucket_name
def get_matching_key_names(bucket_name, key_pattern, s3_cxn = None, output=True):
  ks = get_matching_keys(bucket_name, key_pattern, s3_cxn = s3_cxn, output=output)
  return [k.name for k in ks]
  
def set_s3_key_contents(key, filename, header = None):
  assert key.key is not None
  # retry since sometimes boto fails for mysterious reasons
  try:
    key.set_contents_from_filename(filename)
  except SSLError:
    # wait a second and then retry
    time.sleep(1)
    key.set_contents_from_filename(filename)
   
  if header:     
    for k,v in header.items():
      key.set_metadata(k, v)

def get_s3_key_contents(key, filename):
  assert key.key is not None
  
  try:
    key.get_contents_to_filename(filename)
    # retry if communication times out
  except SSLError:
    time.sleep(1)
    key.get_contents_to_filename(filename)

def download_file_from_s3(bucket_name, key_name, s3_cxn = None, 
  debug=True, overwrite=False):
  tempdir = scratch_dir()
  if debug:
    print "Using", tempdir, "for local storage"
    
  full_filename = os.path.join(tempdir, key_name)
  
  in_key = get_key(bucket_name, key_name)
  if debug:
    print "Downloading", key_name, "to", full_filename, "..."
  if not overwrite and os.path.exists(full_filename) and \
     os.path.getsize(full_filename) == in_key.size:
    if debug:
      print "Already downloaded", full_filename, "from S3"
  else:
    get_s3_key_contents(in_key, full_filename)
  return full_filename

def download_hdf_from_s3(bucket_name, key_name, s3_cxn = None):
  filename = download_file_from_s3(bucket_name, key_name, s3_cxn = s3_cxn)
  return h5py.File(filename)
  
def upload_file_to_s3(filename, bucket_name, key_name, header = {}, s3_cxn = None, debug=True):
  if s3_cxn is None:
    s3_cxn = get_s3_cxn()
  
  try:
    bucket_exists = bucket_name in s3_cxn
  except SSLError:
    time.sleep(1)
    bucket_exists = bucket_name in s3_cxn
  
  if bucket_exists:
    try:
      bucket = s3_cxn.get_bucket(bucket_name)
    except SSLError:
      time.sleep(1)
      bucket = s3_cxn.get_bucket(bucket_name)
  else:
    try:
      bucket = s3_cxn.create_bucket(bucket_name)
    except:
      # catch anything since I've seen weird conflicts while creating
      # a bucket from two places at once 

      time.sleep(1 + random.random()) 
      bucket = s3_cxn.create_bucket(bucket_name)
   
  try:
    key = bucket.get_key(key_name)
  except SSLError:
    key = bucket.get_key(key_name)
  
  if key is None:
    if debug:
      print "Key name", key_name, "not found, creating new key" 
    try:
      key = boto.s3.key.Key(bucket)
    except:
      # sleep for a random amount since not sure if creating a key
      # can create a concurrency conflict AND I DON'T TRUST S3
      r = random.random()
      if debug:
        print "S3 error, backing off for ", (1+r), "seconds and then retrying..."  
      time.sleep(1+r)
      key = boto.s3.key.Key(bucket)
      
  if key is None:
    raise RuntimeError("Couldn't create new key in bucket" + bucket_name)
  key.key = key_name 
  set_s3_key_contents(key, filename, header)
  
from hdf import same_features
def hdf_already_on_s3(bucket_name, key_name,  required_features, s3_cxn = None, 
  debug = True):
  
  if s3_cxn is None:
    s3_cxn = get_s3_cxn()

  try:
    bucket = s3_cxn.get_bucket(bucket_name)
  except SSLError: 
    bucket = s3_cxn.get_bucket(bucket_name)

  if not bucket: 
    if debug:
      print "Couldn't find bucket ", bucket
    return False
    
  try:
    key = bucket.get_key(key_name)
  except SSLError:
    key = bucket.get_key(key_name)
  if key is None:
    return False

  if debug:  
    print "Found", bucket, "/", key, "already on S3"

  features = key.get_metadata('features')
  if features is None:
    print "Missing features attribute"
    return False

  return same_features(features, required_features)
