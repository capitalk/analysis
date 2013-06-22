from optparse import OptionParser
import query 
import cross_info
import query_helpers

def cross_durations(hdf):
  ccy = hdf.attrs['ccy']
  durs = [cross.dur for cross in cross_info.find_crossed_markets_in_hdf(hdf)]
  return ccy, durs   
    
def combine(all_durs, (ccy,durs)):
  all_durs.setsdefault(ccy, []).extend(durs)
  
parser = OptionParser(usage = "usage: %prog s3://bucket-name/key-pattern")
if __name__ == '__main__':
  (options, args) = parser.parse_args()
  assert len(args)==1
  df = query.run(args[0], map_hdf = cross_durations, 
   init = {}, combine = combine, 
   post_process = query_helpers.summarize_continuous)
  print df
