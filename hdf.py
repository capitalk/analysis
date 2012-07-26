import h5py
import pandas
import os 

def add_col(hdf, name, data, compression  = 'lzf'):
  parts = name.split('/')
  dirpath = '/'.join(parts[:-1])
  if len(dirpath) > 0 and dirpath not in hdf:
    hdf.create_group(dirpath)
  hdf.create_dataset(name, 
    data=data, 
    dtype=data.dtype, 
    compression=compression, 
    chunks=True)


def dict_to_hdf(data, path, header, feature_names = None):
  try:
    os.remove(path)
  except OSError:
    pass

  hdf = h5py.File(path, 'w')
  if feature_names is None:
    print "[dict_to_hdf] No feature names given, using dict keys"
    feature_names = data.keys()
    
  hdf.attrs['features'] = feature_names
  ccy1 = header['ccy'][0]
  ccy2 = header['ccy'][1]

  hdf.attrs['ccy1'] = ccy1.encode('ascii')
  hdf.attrs['ccy2'] = ccy2.encode('ascii')
  hdf.attrs['ccy'] = (ccy1 + "/" + ccy2).encode('ascii')
  hdf.attrs['year'] = header['year']
  hdf.attrs['month'] = header['month']
  hdf.attrs['day'] = header['day']
  hdf.attrs['venue'] = header['venue'].encode('ascii')
  hdf.attrs['start_time'] = data['t'][0]
  hdf.attrs['end_time'] = data['t'][-1]

  for name, vec in data.items(): 
    add_col(hdf, name, vec)

  # if program quits before this flag is added, ok to overwrite 
  # file in the future
  hdf.attrs['finished'] = True 
  hdf.close()

def header_from_hdf_filename(filename):
  f = h5py.File(filename)
  a = f.attrs
  
  assert 'ccy1' in a
  assert 'ccy2' in a
  assert 'year' in a
  assert 'month' in a
  assert 'day' in a
  assert 'venue' in a
  assert 'start_time' in a
  assert 'end_time' in a
  assert 'features' in a
  
  header = {
    'ccy': (a['ccy1'], a['ccy2']), 
    'year' : a['year'],
    'month' : a['month'], 
    'day' : a['day'], 
    'venue' : a['venue'], 
    'start_time' : a['start_time'],
    'end_time' : a['end_time'],
    'features': a['features'], 
  }
  f.close()
  return header
  
import numpy as np
def load_from_hdf(path):
  f = h5py.File(path)
  cols = dict([(k,np.array(v[:])) for k, v in f.items()])
  return pandas.DataFrame(data=cols, index=f['t'], dtype='float')  
