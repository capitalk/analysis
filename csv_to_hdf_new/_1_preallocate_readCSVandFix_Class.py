import sys
import time
import pandas as pd
import numpy as np
import argparse
import os
from cStringIO import StringIO
import glob

############# FAST! 752 seconds for EUUS###########################
class capitalKDataFrame:
    '''
    This going to be the Data Structure responsible for storing and indexing the order book information in memory,
    as we read them from the raw CSV files and is going to serve as an intermediate layer between the raw input
    and the hdf buckets that we are going to store in the disk.
    
    The advantage of this structure is that it can dynamically alter its scheme and perform calculations and statistics on the data
    in a very efficient way (as stated in the documentation of pandas/numpy libraries).
    
    Below you can see the very basic scheme of this structure so you can get an idea of what is going to be consisted of and how
    you can build on it and add your personal uber-features.
    
                                A                                                                                       B                                                                                 
                                p                                           v                                           p                                            v                                    
                                1       2        3        4        5        1        2        3        4        5       1        2        3        4        5        1        2        3        4        5
    2012-08-08 00:00:45.442000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  4180000  2000000  0.7932  0.79319  0.79318  0.79317  0.79316   500000  2000000  1000000  2500000  3500000
    2012-08-08 00:00:45.641000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  4180000  2000000  0.7932  0.79319  0.79318  0.79317  0.79316   500000  2000000  1000000  2500000  3000000
    2012-08-08 00:00:45.844000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  5180000  1000000  0.7932  0.79319  0.79318  0.79317  0.79316   500000  2000000  1000000  1500000  4000000
    2012-08-08 00:00:47.268000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  5180000  1000000  0.7932  0.79319  0.79317  0.79316  0.79315   500000  2000000  2500000  4000000  1000000
    2012-08-08 00:00:47.573000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  5180000  1000000  0.7932  0.79319  0.79318  0.79317  0.79316   500000  2000000  1000000  1500000  4000000
    2012-08-08 00:00:48.287000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  5180000  1000000  0.7932  0.79319  0.79317  0.79316  0.79315   500000  2000000  2500000  4000000  1000000
    2012-08-08 00:00:51.241000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  5180000  1000000  0.7932  0.79319  0.79318  0.79317  0.79316   500000  2000000  1000000  1500000  4000000
    2012-08-08 00:00:51.651000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000  1500000   500000  4180000  2000000  0.7932  0.79319  0.79317  0.79316  0.79315   500000  3000000  2500000  3000000  1000000
    2012-08-08 00:00:52.063000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000   500000  1500000  2180000  3000000  0.7932  0.79319  0.79317  0.79316  0.79315  1500000  2000000  4500000  1000000  1000000
    2012-08-08 00:00:52.165000  0.79339  0.7934  0.79341  0.79343  0.79344  1000000   500000  1500000  2180000  4000000  0.7932  0.79319  0.79318  0.79317  0.79316  1500000  1000000  1000000  4500000  1000000
    '''    

    depthBid =None
    depthAsk =None
    biddf = None
    askdf = None
    maxDepth = None
    maxDepthStr = None
    capitalKDF=None
    numOB= None
    pathCSV= None
    numberOfAttrs= None

    def __init__(self, path, colIndexLevels=None, maxDep=5, testMode = False, testNumRows=100):
        
        self.pathCSV = path
        self.depthBid =0
        self.depthAsk =0
        self.biddf = pd.DataFrame( index=["BidVol","BidPrice"])
        self.askdf = pd.DataFrame( index=["AskVol","AskPrice"])
        self.maxDepth = maxDep
        self.maxDepthStr = map(str, np.arange(self.maxDepth)+1)
        self.testmode=testMode
        self.testNumOBs=testNumRows
        
        l1=['A','B']
        l2=['p','v']
        l3=self.maxDepthStr 
        lTot = [l1,l2,l3]
        col_index = pd.MultiIndex.from_tuples( self.generateMultiIndex(lTot) )
        self.numberOfAttrs = len(col_index)
        self.numOB = self.findHowManyOBNeeded(self.pathCSV) #find number of order books to be saved in order to pre-allocate space.
        
        # As explained the index of the rows is going to be the respective timestamp of each order book.
        # Although it would be optimal to know the time stamps a-priori, this can't hold so there are two ways to get round it.
            # 1) pre-allocate the space with non-date indexes, parse the file, save the time stamps and in the end re-name the labels of the indexes.
            # 2) parse the file, crop/strip/ each row in order to find which the date field is and form an index based on this.
        # We preferred the first method as the tokenization of each line is something we can't avoid later on (so as to get the field
        # of prices, volume etc) and thus we selected to incorporate this procedure in the main loop of the program where we read the file and in
        # the end re-index the database with the correct indexes (which is a procedure that doesn't take long), before saving it to HDF. 
        if not testMode:
            self.capitalKDF = pd.DataFrame( -np.ones( (self.numOB, self.numberOfAttrs) ),index=np.arange(self.numOB) , columns=col_index)
        else:
            self.capitalKDF = pd.DataFrame( -np.ones( (testNumRows,self.numberOfAttrs) ),index=np.arange(testNumRows) , columns=col_index)
        
        

    def generateMultiIndex(self, arrays, out=None):
        '''If you want to have a three level multi-index you have to explicitly define all the combinations between the indexes you may have
        i.e. let's say you want to have an index A(sk), pointing to two fields P(rice) and V(olume) and each one of them pointing to
        a number of 5 depth levels.
        Then you need to generate all the possible tuples ('A','v','1'), ('A','v','2'), ('A','v','3') ... ('A','p','1') ...
        This method saves you from the trouble and does everything automatically by using the numpy's array objects to speed up the process.
    
        In other words we just generate a cartesian product of input arrays.
    
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
    
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
    
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])
        '''
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)       
        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.generateMultiIndex(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        out = [ tuple(m) for m in out]
        return out
    def findHowManyOBNeeded(self,path):
        countOB = 0
        with open(path) as f:
            for line in f.readlines():
                if(line.startswith('O')): # +1 for the line rows as if the line starts with O(B) then we should store the order book.
                    countOB+=1
                # if there is a restart we normally should subtract a line but it is not so simple to do it because we have to determine
                # where the RESTART message occurred and if it had any effects on the order preceding order books. Thus we deal with this matter in another method.
        return countOB
    
    def bidAddition(self, newPrice, newVol):
        flag = False
        newTuple = np.array([int(newVol),float(newPrice)])    
        
        if( self.biddf.count(1)[0] ==0 ): #meaning an empty dataframe
            self.biddf['tmp']=newTuple
            self.depthBid = 1
        else:
            for i,v in enumerate(self.biddf.ix['BidPrice']):
                if( v ==newPrice ):
                    self.biddf.ix['BidVol',i] = int(newVol)            
                    flag = True
                    break 
                    # replacement
                    #biddf.ix['BidVol',i] = newVol   
                elif(v<newPrice):
                    if( self.depthBid <  self.maxDepth): self.depthBid += 1
                    self.biddf.insert(i,'tmp',newTuple)
                    flag = True
                    break
                
            if(flag == False and self.depthBid<  self.maxDepth):
                self.depthBid += 1
                self.biddf["tmp"] = newTuple
                
        self.biddf = self.biddf.ix[:,0:self.depthBid]
        self.biddf.columns =  self.maxDepthStr if(self.depthBid == self.maxDepth) else map(str, np.arange(self.depthBid)+1)
       
    def askAddition(self, newPrice, newVol):
        flag = False
        newTuple = np.array([newVol,newPrice])    
        
        if( self.askdf.count(1)[0] ==0 ): #meaning an empty dataframe
            self.askdf['tmp']=newTuple
            self.depthAsk = 1
        else:
            for i,v in enumerate(self.askdf.ix['AskPrice']):
                if( v ==newPrice ):
                    self.askdf.ix['AskVol',i] = newVol            
                    flag = True
                    break 
                elif(v>newPrice):
                    if( self.depthAsk <  self.maxDepth): self.depthAsk += 1
                    self.askdf.insert(i,'tmp',newTuple)
                    flag = True
                    break
                
            if( flag==False and self.depthAsk<  self.maxDepth):
                self.depthAsk += 1
                self.askdf["tmp"] = newTuple
    
        self.askdf = self.askdf.ix[:,0:self.depthAsk]
        self.askdf.columns =  self.maxDepthStr if(self.depthAsk ==  self.maxDepth) else map(str, np.arange(self.depthAsk)+1)
                
    def bidDeletion(self, priceToDelete):
        
        for i,v in enumerate(self.biddf.ix['BidPrice']):
            if( v == priceToDelete ): 
                self.biddf.pop( str(i+1) )
                self.depthBid -=1
                self.biddf = self.biddf.ix[:,0:self.depthBid]
                self.biddf.columns = map(str, np.arange(self.depthBid)+1)
                break
                      
    def askDeletion(self, priceToDelete):
    
        for i,v in enumerate(self.askdf.ix['AskPrice']):
            if( v == priceToDelete ):  
                self.askdf.pop( str(i+1) )
                self.depthAsk -=1
                self.askdf = self.askdf.ix[:,0:self.depthAsk]
                self.askdf.columns = map(str, np.arange(self.depthAsk)+1)
                break

    def parseCSV(self, pathHDF,date):
        
        dateElements = date.split("_")
        dateInDateTimeFormat = pd.datetime( int(dateElements[0]), int(dateElements[1]),int(dateElements[2]) )
        # ----- crop them from the title of the file!

        f= open(self.pathCSV)
        firstLine = True # The first line is just the title.. We don't really want to process it.
        restartFlag = False # Restart is a big issue so we have to flag it, so buckle up and follow the restartFlag.
        i=0
        self.testmode = False
        self.testNumOBs = 100
        tsIndexList = [] # List to hold the time stamps in order to later re-index capitalK Data Frame with the correct time indexes.
        
        prevTimestamp=None
        
        start_time = time.time() #Let's keep track of the time this loop-y pal underneath will take to execute.
        while 1:
            line = f.readline()
            if line =="": break
            if( line.startswith('Q')):
                continue
            
            s = StringIO() # string buffer to read the messages and determine if a RESTART has occured.
            if(firstLine == True): #The header is a specific case where we read a line.
                assert(line.startswith("V3")) # File must be in version V.0.0.3
                firstLine = False
                line = f.readline() #move on to the next line
            
            while not (line.startswith('OB') ):
                if "RESTART" in line:
                    restartFlag = True
                    break
                s.write(line)                    
                line = f.readline()   
                if(line == ""): break # because the file might enter before a message gets printed
            
            if line.startswith('OB'):
                s.write(line) #This line is the OB line so we write it in order to understand later when to print the OB!
            s.seek(0)
            
            while 1:
                if(restartFlag): #if there is a restart message we don't alter the orderbook and not process any messages that we read in the buffer.
                    restartFlag = False
                    break
                line = s.readline() #otherwise we read each line of the buffer (which is the normal message from the exchange).
                if line=="": break    
                if ( line.startswith('A') or line.startswith('D') or line.startswith('OB')):
                    row = line.split(',') 
        
                    if not (len(row) == 4 or len(row) == 5): continue #there may be some half finished line in the file. With the case of length==4 we assure that even if the error happens in the message line we are ok because we don't use that time stamp. (Explain it better if needed).
                    if row[-1]=="": continue # for example: FXCM_EURGBP_2012_07_11.csv, last line
                    if( row[0] == 'A'):
                        if( row[1] == '0'): #its a bid message.
                            self.bidAddition(float(row[3]), float(row[2]) )
                        elif(row[1] == '1'): #its an ask message.
                            self.askAddition(float(row[3]), float(row[2]) )
                        else: sys.exit( "Must Not Happen!" )
                    elif( row[0] == 'D'):
                        if( row[1] == '0'): #its a bid message.
                            self.bidDeletion(float(row[3]))
                        elif(row[1] == '1'): #its an ask message.
                            self.askDeletion(float(row[3]) )
                        else: sys.exit( "Must Not Happen!" )            
                    elif( row[0] == 'OB'):
                        # Some TimeStamps have less digits than 9, so in this variable we save the correct representation of them.
                        #script to check for missing 0 in the nanoseconds position of the time stamp.
                        tsList = row[-1].rstrip('\n').split(':')
                        if len(tsList) == 1:
                            tsList.append('000000000') #i.e. August 08,2012. The header of the last order book is half printed and so we miss the time stamp's field after the ":".
                        elif len(tsList[1]) < 9:
                            tsList[1] = tsList[1].rjust(9,'0')
                        elif len(tsList[1]) > 9: sys.exit("Number of nanoseconds is greater than 9. :-S ")
                        finalTS = pd.Timestamp(int("".join(tsList)))
                        
                        # Good sanity check to check that at least nothing went wrong in the date.
                        if finalTS.date() != dateInDateTimeFormat.date(): sys.exit("Encountered wrong time stamp: "+finalTS)                
                        # end of checking for missing zeros in the beginning. finalTS has the time stamp in the correct format.    
                        
                        # We may have duplicate time stamps in the raw data. In order to deal with this matter we first save the previous
                        # date and if it is the same as the one we examine in presently we decrease the pointer indexes where the next
                        # order book should be added in order to overwrite the previous order book in that position. This way we only
                        # keep the most updated order book for that time stamp. 
                        if finalTS == prevTimestamp:
                            i-=1
                        else:
                            tsIndexList.append( finalTS )
                        prevTimestamp = finalTS # storing the present date. (I know it only relates to the else case but what the heck..)          
                        
                        '''
                        Here is the point where we write stuff to files. If you want to alter the format of the output file this
                        is what you have to mess with! 
                        '''
                        # It's maybe the 200th time that I changed the output format, so I am going to explain it more in the documentation.
                        
                        #1) normally read and append the data structure -- more than the age of the universe!! (more than 40 minutes)
                        #2) pre-allocate space and then change it by changing the index names in every iteration.
                            # no can do baby doll! you can't change only one index with pandas so you have to re-index the whole thing
                            # which is far more naive than re-index them once and for all in the end.
                        
                        #3) pre-process the file to find the dates, pre-allocate space with the exact indexes and then start filling.
                        
                        #4) pre-allocate space with no indexes specified, fill all the values while storing the time stamps,
                        #   re-index the data structure. -- 753 sec = 12.63 minutes
                        
                        self.capitalKDF.ix[i,0:self.depthAsk] = self.askdf.ix['AskPrice'].values
                        self.capitalKDF.ix[i,self.maxDepth:self.maxDepth+self.depthAsk] = self.askdf.ix['AskVol'].values
                        self.capitalKDF.ix[i,2*self.maxDepth:2*self.maxDepth+self.depthBid] = self.biddf.ix['BidPrice'].values
                        self.capitalKDF.ix[i,3*self.maxDepth:3*self.maxDepth+self.depthBid] = self.biddf.ix['BidVol'].values
                        i+=1
                        
                        # One way someone can do it -- Write every row of each dataframe so you can have lists and cols like an ordinary ob.
                        #csvWriter.writerow(["OB",row[1],row[2], finTS] ) # Write the orderbook's header.
                        #for i in range(biddf.count(1)[0]):
                        #    csvWriter.writerow(["Q",row[2],'0',i+1, "%f"%biddf.ix[1,i] , int(biddf.ix[0,i]), int(biddf.ix[0,i]) ])
                        #for i in range(askdf.count(1)[0]):
                        #    csvWriter.writerow(["Q",row[2],'1',i+1, "%f"%askdf.ix[1,i] , int(askdf.ix[0,i]), int(askdf.ix[0,i]) ])
                        ''' 
                        End of writing to file/output
                        '''
                    else: sys.exit("Unknown Input!")#just in case :) 
        
            if self.testmode and i== self.testNumOBs:
                break
        
        rowDifference = len(self.capitalKDF.index) - len(tsIndexList)# we drop the excess lines in case of a restart because we pre-allocated more rows than needed. 
        if( rowDifference != 0 ):
            self.capitalKDF = self.capitalKDF.drop( self.capitalKDF.index[-rowDifference:] )
        
        self.capitalKDF.set_index(keys=pd.DatetimeIndex(tsIndexList), inplace=True )        
        store = pd.HDFStore(pathHDF)
        store.put('capitalKDF',self.capitalKDF)
        store.close()
        print time.time() - start_time, "seconds"
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse CSVs,re-create order books and output HDFs')
    parser.add_argument('-i','--inputPath', help='Path of the raw .CSV files.',dest='in', required=True)
    parser.add_argument('-o','--outputPath', help='Path of the directory to store the HDF buckets.',dest='out', required=True)
    parser.add_argument('-d','--maxDepth', help='Maximum depth of the order book.',dest='depth', required=False)
    parser.add_argument('-t','--testMode', help='Test mode, to stop execution after certain lines and check if everything goes as planned.',dest='test', required=False)
    parser.add_argument('-tl','--testLines', help='Number of order book to stop execution in order to check if all the previous one have the correct values.',dest='testLines', required=False)
    # I have disabled the test mode because I have tested it for now.
    args = vars(parser.parse_args())    
    
    if not os.path.exists(args['out']):
        os.makedirs(args['out'])
    os.chdir(args['in'])
    csvDir = glob.glob("*.csv")
    date= csvDir[0][-14:-4] # this is the date of every pair we are examining.
    
    print "Printing some sanity checks"
    for i in range(len(csvDir)) :
    
        if args['depth']!=None:
            c = capitalKDataFrame( args['in']+csvDir[i],maxDep=args['depth'] )
        else:
            c = capitalKDataFrame( args['in']+csvDir[i] )
        
        print c.biddf.to_string()
        print c.askdf.to_string()
        print "------------------------"
        c.parseCSV(args['out']+csvDir[i][:-4]+".h5", date)
        print c.biddf.to_string()
        print c.askdf.to_string()
        print "============================="
    
