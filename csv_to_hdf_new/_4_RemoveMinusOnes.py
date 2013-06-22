import pandas as pd
import numpy as np
import os, time,argparse
import glob as g

''' This module is responsible for removing any -1 we may find in the prices we are examining.
    In particular if a -1 price is found in place i, we replace the contents of the price and volume of position i
    with those of position -1
    
    Presently we are doing it only for the best bid/ask. I can't really be sure if it would be more correct to discard
    these values or replace them with a valid one from -1, this is why I placed it in a different module so as to study it
    more in the future.''' 

#inputFile = '/home/elnio/Desktop/december7_5sAggreatedEUR/'
#outputFile = '/home/elnio/Desktop/CLEAN_december7_5sAggreatedEUR/'
if __name__ == "__main__":
    start_time = time.time() #Let's keep track of the time this loop-y pal underneath will take to execute.
    
    parser = argparse.ArgumentParser(description='Read HDFs and clean any -1s you may find in the best bid or ask prices.')
    parser.add_argument('-i','--inputPath', help='Path of the aligned HDFs.',dest='in', required=True)
    parser.add_argument('-o','--outputPath', help='Path of the directory to store the aggregated HDF buckets.',dest='out', required=True)
    args = vars(parser.parse_args())    
 
    if not os.path.exists(args['out']):
        os.makedirs(args['out'])
    os.chdir(args['in'])
    hdfsDir = g.glob("*.h5")
    hdfs=[]
    hdfsDFfromIndexes=[]

    for bucketName in hdfsDir:
        buck = pd.read_hdf(args['in']+bucketName,'capitalKDF')
        minus1PositionsAsk = np.nonzero( buck.A.p['1'] < 0 )[0]
        if len(minus1PositionsAsk)>0:print "Found",len(minus1PositionsAsk), "-1s in the best ask price of",bucketName
        
        for pos in minus1PositionsAsk:
            buck.A.p['1'][pos] = buck.A.p['1'][pos-1]
            buck.A.v['1'][pos] = buck.A.v['1'][pos-1]
        
        minus1PositionsBid = np.nonzero( buck.B.p['1'] < 0 )[0]
        if len(minus1PositionsBid)>0:print "Found",len(minus1PositionsBid), "-1s in the best bid price of",bucketName
        
        for pos in minus1PositionsBid:
            buck.B.p['1'][pos] = buck.B.p['1'][pos-1]
            buck.B.v['1'][pos] = buck.B.v['1'][pos-1]
    
        
        store = pd.HDFStore(args['out']+bucketName)
        store.put('capitalKDF',buck)
        store.close()
  
    print time.time() - start_time, "seconds"     
