import pandas as pd
import os
import glob
import argparse,time

#inputFile='/home/elnio/Desktop/december7MYHDFs/'
#outputFile = '/home/elnio/Desktop/onlyEURdecember7AlignedBuckets/'

if __name__ == "__main__":
    start_time = time.time() #Let's keep track of the time this loop-y pal underneath will take to execute.
    
    parser = argparse.ArgumentParser(description='Read HDFs with sparse ticks and align them so they all have the same time scale')
    parser.add_argument('-i','--inputPath', help='Path of the HDFs with the sparse ticks.',dest='in', required=True)
    parser.add_argument('-o','--outputPath', help='Path of the directory to store the aligned HDF buckets.',dest='out', required=True)
    parser.add_argument('-p','--pairs', help='If you want to select only the EUR pairs, input EUR.Likewise for all the other pairs. Leave blank if you want to align all the currency pairs.',dest='p', required=False)
    args = vars(parser.parse_args())    


    if not os.path.exists(args['out']):
        os.makedirs(args['out'])
    os.chdir(args['in'])
    hdfs=[]
    hdfsDFfromIndexes=[]
    if args['p'] != None:
        hdfsDir = glob.glob("FXCM_"+args['p']+"*.h5") # THIS MEANS WE ARE ONLY GOING TO ALIGN THE PAIRS YOU SPECIFY.
    else:
        hdfsDir = glob.glob("*.h5") # THIS MEANS WE ARE ONLY GOING TO ALIGN ALL THE PAIRS.
    
    for i in range( len(hdfsDir)):
        buck = pd.read_hdf(hdfsDir[i],'capitalKDF')
        hdfs.append(buck)
        hdfsDFfromIndexes.append( pd.DataFrame(index = buck.index) )
    
    totIndexes = pd.concat( hdfsDFfromIndexes, axis=1).index
    
    print "We will now print the shapes of the new data frames. They all ought to be the same."
    
    for i in range( len(hdfs)):
        store = pd.HDFStore(args['out']+hdfsDir[i])
        reindexedBuck = hdfs[i].reindex(totIndexes)
        reindexedBuck = reindexedBuck.fillna(method='pad') # propagation of the last valid value -- filling missing order books so everything is aligned.
        reindexedBuck = reindexedBuck.fillna(method='bfill') # back propagation -- filling the first line if it still has NA.
        store.put('capitalKDF',reindexedBuck)
        store.close()
        
        print reindexedBuck.shape
        
    print "Alignment process finished successfully."
    print time.time() - start_time, "seconds"