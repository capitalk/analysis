
import boto 
import cloud 

picloud_id = 2579
picloud_secret_key = 'f228c0325cf687779264a0b0698b0cfe40148d65'
cloud.setkey(picloud_id, picloud_secret_key)


def get_s3_cxn():
  aws_access_key = 'AKIAJSCF3K3HKREPYE6Q'
  aws_secret_key = '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'
  s3_cxn = boto.connect_s3(aws_access_key, aws_secret_key)
  if s3_cxn is None:
    raise RuntimeError("Couldn't connect to S3")
  else:
    return s3_cxn

import time 
from ssl import SSLError 
def set_s3_key_contents(key, filename, header = None):
  # retry since sometimes boto fails for mysterious reasons
  try:
    key.set_contents_from_filename(filename)
  except SSLError:
    time.sleep(1)
    key.set_contents_from_filename(filename)
   
  if header:     
    for k,v in header.items():
      key.set_metadata(k, v)


def get_s3_key_contents(key, filename):
  try:
    key.get_contents_to_filename(filename)
    # retry if communication times out
  except SSLError:
    time.sleep(1)
    key.get_contents_to_filename(filename)
