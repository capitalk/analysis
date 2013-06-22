from optparse import OptionParser
import numpy as np
import query 
import pandas

def count_crosses(hdf):
  ccy = hdf.attrs['ccy']
  date = (hdf.attrs['year'], hdf.attrs['month'], hdf.attrs['day'])
  count = np.sum(hdf['bid'][:] > hdf['offer'][:])
  print "Found", count, "crossed tick for", ccy, "on", date
  return ccy, count

def combine(counts, (ccy, count)):
  old = counts.get(ccy,0)
  counts[ccy] = old + count
  
def convert_to_dataframe(total_counts):
  return pandas.DataFrame({"count":total_counts.values()}, index=total_counts.keys())

parser = OptionParser(usage = "usage: %prog s3://bucket-name/key-pattern")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  assert len(args)==1
  df = query.run(args[0], map_hdf = count_crosses, 
   init = {}, combine = combine, post_process = convert_to_dataframe)
  print df
  