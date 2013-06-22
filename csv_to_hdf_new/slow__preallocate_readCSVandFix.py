import sys
import time
from cStringIO import StringIO
import numpy as np
import pandas as pd


''' Changed the preallocate_readCSVandFix module so as to read from the cells of the total dataframe and apply those
changes without the need of biddf and askdf. It should be faster but instead its much much slower.'''

################################### SLOW!!! 1984 seconds for EUUS #################################3
class capitalKDataFrame:
    '''
    This going to be the Data Structure responsible for storing and indexing the order book information in memory,
    as we read them from the raw CSV files and is going to serve as an intermediate layer between the raw input
    and the hdf buckets that we are going to store in the disk.
    
    The advantage of this structure is that it can dynamically alter its scheme and perform calculations and statistics on the data
    in a very efficient way (as stated in the documentation of pandas/numpy libraries).
    
    Below you can see the very basic scheme of this structure so you can get an idea of what is going to be consisted of and how
    you can build on it and add your personal uber-features.
    
    (!!!! put here a print of the dataframe so you won't design it by hand
         and explain all about the indexes and everything. )
    '''    
    
    maxDepth = 5
    numberOfAttrs= 20
    maxDepthStr = map(str, np.arange(maxDepth)+1)
    capitalKDF=None
    numOB= None
    pathCSV= None
    l1=['A','B']
    l2=['p','v']
    l3=maxDepthStr 
    lTot = [l1,l2,l3]

    def __init__(self, path, colIndexLevels=lTot, testMode = False, testNumRows=0):
        self.pathCSV = path
        col_index = pd.MultiIndex.from_tuples( self.generateMultiIndex(self.lTot) )
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

    def bidAddition(self, row, newPrice, newVol):
        col_position = None
        # determine the position to add the new value
        posEqual = np.nonzero( newPrice == self.capitalKDF.ix[row,('B','p')] )[0] #this will turn a list of positions (hopefully only one because we are talking about equality).
        if len(posEqual) != 0:
            self.capitalKDF.ix[row,('B','v')][posEqual] = newVol
        else: 
            posLessThan = np.nonzero( newPrice > self.capitalKDF.ix[row,('B','p')] )[0]
            if len(posLessThan) != 0: # as the array is filled in the beginning with -1, the above code will return an empty list even if there are empty position to fill. Thus we have to investigate it even more...
                col_position = posLessThan[0] #the first position is the one that we want to insert the new element
                self.capitalKDF.ix[row,('B','p')][col_position+1:] = self.capitalKDF.ix[row,('B','p')][col_position:-1]
                self.capitalKDF.ix[row,('B','p')][col_position] = newPrice
                self.capitalKDF.ix[row,('B','v')][col_position+1:] = self.capitalKDF.ix[row,('B','v')][col_position:-1]
                self.capitalKDF.ix[row,('B','v')][col_position] = newVol
        
    def askAddition(self, row, newPrice, newVol):
        col_position = None
        # determine the position to add the new value
        posEqual = np.nonzero( newPrice == self.capitalKDF.ix[row,('A','p')] )[0] #this will turn a list of positions (hopefully only one because we are talking about equality).
        if len(posEqual) != 0: # modification message.
            self.capitalKDF.ix[row,('A','v')][posEqual] = newVol #in this case we only need to change the respective volume of the price as we are talking about an addition/modification message for this particular exchange.
        else:
            posGreaterThan = np.nonzero( newPrice < self.capitalKDF.ix[row,('A','p')] )[0]
            if len(posGreaterThan) != 0:#in the case of an addition in the ask prices we need to check if there are unfilled places with -1 as the less than condition does not assure that the empty positions will be filled.
                col_position = posGreaterThan[0] #the first position is the one that we want to insert the new element
            else: 
                posDeeperDepth = np.nonzero(self.capitalKDF.ix[row,('A','p')] ==-1 )[0]
                if len(posDeeperDepth!=0):
                    col_position = posDeeperDepth[0]
            if col_position != None:
                self.capitalKDF.ix[row,('A','p')][col_position+1:] = self.capitalKDF.ix[row,('A','p')][col_position:-1]
                self.capitalKDF.ix[row,('A','p')][col_position] = newPrice
                self.capitalKDF.ix[row,('A','v')][col_position+1:] = self.capitalKDF.ix[row,('A','v')][col_position:-1]
                self.capitalKDF.ix[row,('A','v')][col_position] = newVol

        

    def bidDeletion(self, row, priceToDelete):
        col_position=None
        posEqual = np.nonzero( priceToDelete == self.capitalKDF.ix[row,('B','p')] )[0]
        if len(posEqual) != 0:
            col_position = posEqual[0]
            self.capitalKDF.ix[row,('B','p')][col_position:-1] = self.capitalKDF.ix[row,('B','p')][col_position+1:]
            self.capitalKDF.ix[row,('B','v')][col_position:-1] = self.capitalKDF.ix[row,('B','v')][col_position+1:]
            
            self.capitalKDF.ix[row,('B','p')][-1]= -1
            self.capitalKDF.ix[row,('B','p')][-1]= -1
            self.capitalKDF.ix[row,('B','v')][-1]= -1
            self.capitalKDF.ix[row,('B','v')][-1]= -1

    def askDeletion(self,row, priceToDelete):
        col_position=None
        posEqual = np.nonzero( priceToDelete == self.capitalKDF.ix[row,('A','p')] )[0]
        if len(posEqual) != 0:
            col_position = posEqual[0]
            self.capitalKDF.ix[row,('A','p')][col_position:-1] = self.capitalKDF.ix[row,('A','p')][col_position+1:]
            self.capitalKDF.ix[row,('A','v')][col_position:-1] = self.capitalKDF.ix[row,('A','v')][col_position+1:]
            
            self.capitalKDF.ix[row,('A','p')][-1]= -1
            self.capitalKDF.ix[row,('A','p')][-1]= -1
            self.capitalKDF.ix[row,('A','v')][-1]= -1
            self.capitalKDF.ix[row,('A','v')][-1]= -1


    def parseCSV(self, pathHDF):
        
        # FOR NOW
        date = "2012_08_08"
        dateElements = date.split("_")
        dateInDateTimeFormat = pd.datetime( int(dateElements[0]), int(dateElements[1]),int(dateElements[2]) )
        # ----- crop them from the title of the file!
        
        
        f= open(self.pathCSV)
        firstLine = True # The first line is just the title.. We don't really want to process it.
        restartFlag = False # Restart is a big issue so we have to flag it, so buckle up and follow the restartFlag.
        i=0
        testmode = False
        testNumOBs = 16

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

                    if not (len(row) == 4 or len(row) == 5): continue #there may be some half finished lines in the file. With the case of length==4 we assure that even if the error happens in the message line we are ok because we don't use that time stamp. (Explain it better if needed).
            
                    if( row[0] == 'A'):
                        if( row[1] == '0'): #its a bid message.
                            self.bidAddition(i, float(row[3]), float(row[2]) )
                        elif(row[1] == '1'): #its an ask message.
                            self.askAddition(i, float(row[3]), float(row[2]) )
                        else: sys.exit( "Must Not Happen!" )
                    elif( row[0] == 'D'):
                        if( row[1] == '0'): #its a bid message.
                            self.bidDeletion(i, float(row[3]))
                        elif(row[1] == '1'): #its an ask message.
                            self.askDeletion(i, float(row[3]) )
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
                            self.capitalKDF.ix[i-1]= self.capitalKDF.ix[i]   
                            continue
                        else:
                            tsIndexList.append( finalTS )
                        prevTimestamp = finalTS # storing the present date. 
                    
                        '''
                        Here is the point where we write stuff to files. If you want to alter the format of the output file this
                        is what you have to mess with! 
                        '''
                        # It's probably the 200th time that I changed the output format, so I am going to explain it more in the documentation.
                        
                        #1) normally read and append the data structure -- more than the age of the universe!! (more than 40 minutes)
                        #2) pre-allocate space and then change it by changing the index names in every iteration.
                            # no can do baby doll! you can't change only one index with pandas so you have to re-index the whole thing
                            # which is far more naive than re-index them once and for all in the end.
                        
                        #3) pre-process the file to find the dates, pre-allocate space with the exact indexes and then start filling.
                        
                        #4) pre-allocate space with no indexes specified, fill all the values while storing the time stamps,
                        #   re-index the data structure. -- 753 sec = 12.63 minutes
                        
                        # One way someone can do it -- Write every row of each dataframe so you can have lists and cols like an ordinary ob.
                        #csvWriter.writerow(["OB",row[1],row[2], finTS] ) # Write the orderbook's header.
                        #for i in range(biddf.count(1)[0]):
                        #    csvWriter.writerow(["Q",row[2],'0',i+1, "%f"%biddf.ix[1,i] , int(biddf.ix[0,i]), int(biddf.ix[0,i]) ])
                        #for i in range(askdf.count(1)[0]):
                        #    csvWriter.writerow(["Q",row[2],'1',i+1, "%f"%askdf.ix[1,i] , int(askdf.ix[0,i]), int(askdf.ix[0,i]) ])
                        ''' 
                        End of writing to file/output
                        '''

                        i+=1
                        if i!=self.numOB: 
                            self.capitalKDF.ix[i]= self.capitalKDF.ix[i-1]              
                    
                else: sys.exit("Unknown Input!")#just in case :) 

            if testmode and i== testNumOBs:
                break
        
        rowDifference = len(self.capitalKDF.index) - len(tsIndexList)# we drop the excess lines in case of a restart because we pre-allocated more rows than needed. 
        if( rowDifference != 0 ):
            self.capitalKDF = self.capitalKDF.drop( self.capitalKDF.index[-rowDifference:] )

        self.capitalKDF.set_index(keys=pd.DatetimeIndex(tsIndexList), inplace=True )
        #print cpk.capitalKDF.to_string()    
        store = pd.HDFStore(pathHDF)
        store.put('capitalKDF',self.capitalKDF)
        store.close()
        print time.time() - start_time, "seconds"
        
c = capitalKDataFrame("/home/elnio/Desktop/raw_august8/FXCM_EURUSD_2012_08_08.csv")
c.parseCSV("/home/elnio/Desktop/mpla.h5")