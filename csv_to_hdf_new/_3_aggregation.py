import pandas as pd
import os
import glob as g
import time
import argparse


def timeAggregation(date):
    #aggregatedTimePoint= (date.second/10)*10 #for now its 10 second frames!
    aggregatedTimePoint= date.second - ((date.second%5))  #for now its 5 second frames!
    #aggregatedTimePoint= date.second - ((date.second%2)) #for now its 3 second frames
    #aggregatedTimePoint= date.second #for now its 1 second frames
    
    newD= pd.datetime(date.year,date.month,date.day,date.hour,date.minute, aggregatedTimePoint)
    return newD    


#inputFile = '/home/elnio/Desktop/onlyEURdecember7AlignedBuckets/'
#outputFile = '/home/elnio/Desktop/december7_5sAggreatedEUR/'

if __name__ == "__main__":
    start_time = time.time() #Let's keep track of the time this loop-y pal underneath will take to execute.
    
    parser = argparse.ArgumentParser(description='Read aligned HDFs and aggregate them in time frames of your choice.')
    parser.add_argument('-i','--inputPath', help='Path of the aligned HDFs.',dest='in', required=True)
    parser.add_argument('-o','--outputPath', help='Path of the directory to store the aggregated HDF buckets.',dest='out', required=True)
    args = vars(parser.parse_args())    
    
    if not os.path.exists(args['out']):
        os.makedirs(args['out'])
    os.chdir(args['in'])
    hdfsDir = g.glob("*.h5")
    hdfs=[]
    for bucketName in hdfsDir:
        buck = pd.read_hdf(args['in']+bucketName,'capitalKDF')
        store = pd.HDFStore(args['out']+bucketName)
        aggregatedBuck = buck.groupby(timeAggregation).mean()
        store.put('capitalKDF',aggregatedBuck)
        store.close()
    
    print time.time() - start_time, "seconds"