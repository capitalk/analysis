from argparse import ArgumentParser 
import query 
import cross_info
import query_helpers

parser = ArgumentParser(description='Process some integers.')
parser.add_argument('pattern', metavar='P', type=str, nargs='1',
                     help='s3://capk-bucket/some-hdf-pattern')
parser.add_argument('--min-duration', dest='min_dur', type=int, default=None, 
  help  = 'ignore crosses which last shorter than this min. duration in milliseconds')


MIN_DUR = None

def cross_amounts(hdf):
  ccy = hdf.attrs['ccy']
  amts = [cross.amt * cross.min_vol \
    for cross in cross_info.find_crossed_markets_in_hdf(hdf) 
    if MIN_DUR is None or cross.dur >= MIN_DUR]
  return ccy, amts   
    
def combine(all_amts, (ccy,amts)):
  all_amts.setdefault(ccy, []).extend(amts)
  
if __name__ == '__main__':
  args = parser.parse_args()
  MIN_DUR = args.min_dur 
  df = query.run(args.pattern, 
    map_hdf = cross_amounts, 
   init = {}, combine = combine, 
   post_process = query_helpers.summarize_dict )
  print df
