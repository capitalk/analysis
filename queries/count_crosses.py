from optparse import OptionParser
import numpy as np
import query 
import pandas

def count_crosses(hdf):
  ccy = hdf.attrs['ccy']
  date = (hdf.attrs['year'], hdf.attrs['month'], hdf.attrs['day'])
  count = np.sum(hdf['bid'][:] > hdf['offer'][:])
  print "Found", count, "crosses for", ccy, "on", date
  return count, ccy

def combine(counts, (count,ccy)):
  if ccy in counts: counts[ccy] += count
  else: counts[ccy] = count
  return counts 

def convert_to_dataframe(total_counts):
  ccys = total_counts.keys()
  df = pandas.DataFrame({"count":total_counts.values()}, index=ccys)
  return df
  
parser = OptionParser(usage = "usage: %prog s3://bucket-name/key-pattern")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  assert len(args)==1
  df = query.run(args[0], map_hdf = count_crosses, 
   init = {}, combine = combine, post_process = convert_to_dataframe)
  print df
  