import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import lag_plot, autocorrelation_plot
import numpy as np
import glob as g
import os
import time
import argparse
import sys

def printDataFrame(pth, bucketName, time=None):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    print "CapitalK Data Frame for ", bucketName
    if time==None:
        print df
    else:
        print eval("df.ix["+time+"].to_string()")
    
def printDataDescription(pth, bucketName):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    print "Summarization of CapitalK Data Frame for ", bucketName
    print df.describe().to_string()
    
def findMaxSpread(pth, bucketName):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    print "Max Spread of CapitalK Data Frame for ", bucketName
    print "You can see the maximum spread for each of the respective price depths"
    print "Depth  Spread"
    print df.max()['A','p'] - df.max()['B','p']
    
def checkForCrossedMarket(pth, bucketName):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    print "Check for Crossed Market of CapitalK Data Frame for ", bucketName
    print "You can see in which positions the bid price is higher than the ask price"
    depth= df['A','p'].shape[1]
    for i in range(depth):
        print "Level",(i+1)
        print np.nonzero(df['A','p',str(i+1)]<df['B','p',str(i+1)])
    
def findCorrelation(pth, bucket1, bucket2, time=None ):
    df1 = pd.read_hdf(pth+bucket1,'capitalKDF')
    df2 = pd.read_hdf(pth+bucket2,'capitalKDF')
    print "Check for correlations between ", bucket1,"and",bucket2+".\n"
    
    if time==None:
        print "Using all records."
        print "Correlation of the ask prices:",df1['A','p','1'].corr(df2['A','p','1'])
        print "Correlation of the bid prices:",df1['B','p','1'].corr(df2['B','p','1'])
    else:
        print "Using", time,"records."
        print "Correlation of the ask prices:", eval( 'df1.ix['+time+',(\'A\',\'p\',\'1\')].corr(df2.ix['+time+',(\'A\',\'p\',\'1\')])' )
        print "Correlation of the ask prices:", eval( 'df1.ix['+time+',(\'B\',\'p\',\'1\')].corr(df2.ix['+time+',(\'B\',\'p\',\'1\')])' )

def plotLag(pth, bucketName):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    plt.subplot(2,2,1)
    lag_plot(df['A','p','1'])
    plt.title("Lag plot for best ask price")
    plt.subplot(2,2,2)
    lag_plot(df['A','v','1'])
    plt.title("Lag plot for best ask volume")
    plt.subplot(2,2,3)
    lag_plot(df['B','p','1'])
    plt.title("Lag plot for best bid price")
    plt.subplot(2,2,4)
    lag_plot(df['B','v','1'])
    plt.title("Lag plot for best bid volume")
    plt.show()
    
def plotAutocorrelation(pth, bucketName):
    df = pd.read_hdf(pth+bucketName,'capitalKDF')
    autocorrelation_plot(df['A','p','1'], plt.subplot(2,2,1))
    plt.title("Lag plot for best ask price")
    autocorrelation_plot(df['A','v','1'], plt.subplot(2,2,2))
    plt.title("Lag plot for best ask volume")
    autocorrelation_plot(df['B','p','1'], plt.subplot(2,2,3))
    plt.title("Lag plot for best bid price")
    autocorrelation_plot(df['B','v','1'], plt.subplot(2,2,4))
    plt.title("Lag plot for best bid volume")
    plt.show()

def plotBBOAllCurrencies(pth):
    os.chdir(pth)
    hdfsDir = g.glob("*.h5")
    i=0
    for bucketName in hdfsDir:
        buck = pd.read_hdf(pth+bucketName,'capitalKDF')
        plt.subplot(7,4,i)
        buck['A','p','1'].plot()        
        buck['B','p','1'].plot()
        plt.title(bucketName[5:11])
        i+=1
    plt.legend()
    plt.show()
        
def histogram(pth, time=None):
    os.chdir(pth)
    hdfsDir = g.glob("*.h5")
    askvolbuck = pd.DataFrame(index = pd.read_hdf(g.glob(pth+"*.h5")[0],'capitalKDF').index)
    bidvolbuck = pd.DataFrame(index = pd.read_hdf(g.glob(pth+"*.h5")[0],'capitalKDF').index)
    for bucketName in hdfsDir:
        askvolbuck[bucketName[5:11]] = pd.read_hdf(pth+bucketName,'capitalKDF')['A','v','1']        
        bidvolbuck[bucketName[5:11]] = pd.read_hdf(pth+bucketName,'capitalKDF')['B','v','1']        

    if time==None:
        askvolbuck.plot(kind='bar')
    else:
        eval("askvolbuck["+time+"].plot(kind='bar')")
    plt.title("Ask Volume")
    if time==None:
        bidvolbuck.plot(kind='bar')
    else:
        eval("bidvolbuck["+time+"].plot(kind='bar')")
    plt.title("Bid Volume")
    plt.show()


if __name__ == "__main__":
    start_time = time.time() #Let's keep track of the time this loop-y pal underneath will take to execute.
    
    parser = argparse.ArgumentParser(description='Demo Methods')
    parser.add_argument('-c','--choice', help='A number between 1-7, matching the method\'s code you want to execute.',
                        dest='choice', required=True, choices=np.arange(9)+1, type=int)
    parser.add_argument('-i','--inputPath', help='Path of the directory where HDF buckets lie.',dest='in', required=True)
    parser.add_argument('-b','--bucketName', help='Name of the HDF bucket you want to use.',dest='buck', required=False)
    parser.add_argument('-t','--time', help='Time horizon to execute on.',dest='t', required=False)
    parser.add_argument('-b2','--bucket2', help='Name of the HDF bucket to correlate with bucket 1.',dest='buck2', required=False)

    args = vars(parser.parse_args())    
    
    if args['choice'] == 1:
        if args['buck'] == None: sys.exit("You should specify a bucket's name to print, using -b option.\nPlease try again.")
        printDataFrame(args['in'], args['buck'], time=args['t'])
    elif args['choice'] == 2:
        printDataDescription(args['in'], args['buck'])
    elif args['choice'] == 3:
        findMaxSpread(args['in'], args['buck'])
    elif args['choice'] == 4:
        checkForCrossedMarket(args['in'], args['buck'])
    elif args['choice'] == 5:
        if args['buck2'] == None: sys.exit("You should specify a bucket's name to correlate with the first bucket you specified.\nPlease try again.")
        findCorrelation(args['in'], args['buck'], args['buck2'], time=args['t'])
    elif args['choice'] == 6:
        plotLag(args['in'], args['buck'])
    elif args['choice'] == 7:
        plotAutocorrelation(args['in'], args['buck'])
    elif args['choice'] == 8:
        plotBBOAllCurrencies(args['in'])
    elif args['choice'] == 9:
        histogram(args['in'],args['t'])
    else:
        print "what the heck this is wrong :S"
    print time.time() - start_time, "seconds"


#filename = "FXCM_EURGBP_2012_08_08.h5"
#pathHDF="/home/elnio/Desktop/10sAggRemovedMinusOnesALL/"

#printDataFrame(pathHDF,filename,time="10:50")
#printDataDescription(pathHDF,filename)
#findMaxSpread(pathHDF,filename)
#checkForCrossedMarket(pathHDF,filename)
#plotBBOAllCurrencies(pathHDF)
#histogram(pathHDF)