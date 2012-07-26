
from optparse import OptionParser
import cloud_helpers
import fnmatch 
def del_keys(bucket, pat):
  print "Finding bad key names..."
  b = cloud_helpers.get_bucket(bucket)
  bad_keys = []
  for key in b:
     if fnmatch.fnmatch(key.name, pat) and not fnmatch.fnmatch(key.name, '*_20??_??_*'):
       print "Marking", key.name, "for deletion"
       bad_keys.append(key)
  if len(bad_keys) == 0:
    print "No bad keys found"
  else: 
    print "About to delete", len(bad_keys), "bad keys."
    confirm = raw_input("Are you sure? [y/N]: ")
    if confirm.lower() in ['yes', 'y']:
      b.delete_keys(bad_keys)
    else:
      print "Deletion canceled"
    
parser = OptionParser(usage = "usage: %prog [options] path")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  
  if len(args) != 1:
    parser.print_help()
  else:
    arg = args[0]
    if arg.startswith("s3://"):
      arg = arg[5:]
    if arg.endswith("/"):
      arg = arg[:-1]
    if "/" in arg:
      bucket, _, pat = arg.partition("/")
    else:
      bucket = arg
      pat = "*"
    del_keys(bucket, pat)