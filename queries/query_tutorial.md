The Capital K Query Framework 
==============================
_(or, a poor man's MapReduce made with picloud and duct tape)_
---------------------------------------------------------------

Very often we want to perform computations which proceed according to the following basic pattern:

  1. Select a subset of the our HDF files from S3 (commonly by using a pattern over their names as a filter)
  2. Distributed the selected S3 keys to worker instances on picloud
  3. Have each picloud worker download its HDF and return some computed quantities (i.e. event counts, probabilities, etc...)
  4. Combine all the workers' returned values into a single final result. 

If you implement a few queries like this from scratch you'll start to notice that the majority of your code will is repetitive plumbing (calls to `cloud`, `boto`, `h5py`, etc...). The *query* module factors out all this plumbing for you, saving you from certain madness and letting you focus on actually learning doing useful with our data. 

Since we're only trying to support a simple use-case (parallel map over HDFs, sequential/local reduce) the query API is extremely simple. In fact it consists of just a single function, which takes several of your functions and weaves them together to perform a query. 

    query.run(bucket, key_pattern = None, 
      map_hdf = None, map_dataframe = None,
      init = None, combine = None,  
      post_process = None, 
      accept_none_as_result = True, 
      label = None, _type = 'f2')

There are quite a few arguments here but the only essential ones are *bucket*, *key_pattern*, *map\_hdf*, *init*, and *combine*. 

Selecting Files
----------------
The first two arguments (*bucket* and *key\_pattern*) tell the query engine which files you'd like to process. The bucket has to be the exact name of one of our S3 buckets (i.e. "capk-fxcm-hdf") but the key pattern can contain wildcards (i.e. "USDJPY\*2012\_07\*" will select all USD/JPY files from July of 2012). You can also, as a convenience, give a complete pattern for *bucket* (i.e. "s3://capk-fxcm/\*EURUSD\*") and leave *key\_pattern* as `None`. 

Parallel Map
----------------
The function you supply as *map_hdf* will be given an HDF and is responsible for returning some sort of computed result. This is the part of your code which will run on picloud and often is the heart of your query. If you want your data loaded into a pandas DataFrame you can instead pass your function to the *map_dataframe* argument, which loads the HDF fields into a DataFrame and then passes that DataFrame to your code. 


Combining Partial Results
--------------------------
As the results of your parallel mappers return asynchronously from picloud, they will be combined into a single value using whatever function you supply as the *combine* argument. This function must take the current aggregated value and a single mapper's result. It can then either return a new aggregated value (if you're feeling Functional) or simply modify the aggregated value. But what is the aggregated value used to "combine" the very first map result? That's the *init* argument. Often, you can make the initial value some mutable object (i.e. `init = []` or `init={}`) and simplify append results as they arrive. 


Example: Min/Max Midprice
----------------------------------
Let's say we wanted to know the extrema of the midprice for some particular currency pair. We can split the work into:
  
  1. a parallel mapper which, for every HDF, computes the midprice and returns its min and max.   
  2. a combiner which simply concatenates all the values into a single list 
  3. after all the work is done we can simply get the min/max values of the combined list. 

We'll get the currency pattern from the commandline arguments, turn it into a wildcard pattern, and then launch a query. 
    import numpy as np 

    def map(hdf):
      """Takes an HDF, returns a tuple of min/max midprices"""
      # Note: I slice into the HDF's column to pull out a numpy array 
      midprice = (hdf['bid'][:]+hdf['bid'][:])/2.0
      return np.min(midprice), np.max(midprice)
    
    def combine(values, (curr_min, curr_max)):
      values.append(curr_min)
      values.append(curr_max)

    
    from argparse import ArgumentParser 
    parser = ArgumentParser(description='min/max for ccy pair')
    parser.add_argument('--ccy', dest='ccy', type=str, required=True, help="e.g. USDJPY")
    parser.add_argument('--bucket', dest='bucket', type=str, default='capk-fxcm-hdf')
    if __name__ == '__main__':
      args = parser.parse_args() 
      pattern = '*' + args.ccy + '*.hdf'
      values = query.run(args.bucket, pattern, 
	    map_hdf = map, combine = combine, init = [])
	  print "Min: %s, Max: %s" % (np.min(values), np.max(values))
	
