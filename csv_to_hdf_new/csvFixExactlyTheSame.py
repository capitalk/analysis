import csv
import numpy as np
import pandas as pd
import time
from cStringIO import StringIO
import argparse
import os

maxDepth = 5
depthBid =0
depthAsk =0
numberOfAttrs= 2
maxDepthStr = map(str, np.arange(maxDepth)+1)

biddf = pd.DataFrame( index=["BidVol","BidPrice"])
askdf = pd.DataFrame( index=["AskVol","AskPrice"])

def bidAddition(newPrice, newVol):
    global depthBid
    global biddf
    flag = False
    newTuple = np.array([int(newVol),float(newPrice)])    
    
    if( biddf.count(1)[0] ==0 ): #meaning an empty dataframe
        biddf['tmp']=newTuple
        depthBid = 1
    else:
        for i,v in enumerate(biddf.ix['BidPrice']):
            if( v ==newPrice ):
                biddf.ix['BidVol',i] = int(newVol)            
                flag = True
                break 
                # replacement
                #biddf.ix['BidVol',i] = newVol   
            elif(v<newPrice):
                if( depthBid < maxDepth): depthBid += 1
                biddf.insert(i,'tmp',newTuple)
                flag = True
                break
            
        if(flag == False and depthBid< maxDepth):
            depthBid += 1
            biddf["tmp"] = newTuple
            
    biddf = biddf.ix[:,0:depthBid]
    biddf.columns = maxDepthStr if(depthBid == maxDepth) else map(str, np.arange(depthBid)+1)
    
def askAddition(newPrice, newVol):
    global depthAsk
    global askdf
    flag = False
    newTuple = np.array([newVol,newPrice])    
    
    if( askdf.count(1)[0] ==0 ): #meaning an empty dataframe
        askdf['tmp']=newTuple
        depthAsk = 1
    else:
        for i,v in enumerate(askdf.ix['AskPrice']):
            if( v ==newPrice ):
                askdf.ix['AskVol',i] = newVol            
                flag = True
                break 
            elif(v>newPrice):
                if( depthAsk < maxDepth): depthAsk += 1
                askdf.insert(i,'tmp',newTuple)
                flag = True
                break
            
        if( flag==False and depthAsk< maxDepth):
            depthAsk += 1
            askdf["tmp"] = newTuple

    askdf = askdf.ix[:,0:depthAsk]
    askdf.columns = maxDepthStr if(depthAsk == maxDepth) else map(str, np.arange(depthAsk)+1)
                
def bidDeletion(priceToDelete, volToDelete):
    global biddf
    global depthBid
    
    for i,v in enumerate(biddf.ix['BidPrice']):
        if( v == priceToDelete ): 
            biddf.pop( str(i+1) )
            depthBid -=1
            biddf = biddf.ix[:,0:depthBid]
            biddf.columns = map(str, np.arange(depthBid)+1)
            break
                      
def askDeletion(priceToDelete, volToDelete):
    global askdf
    global depthAsk

    for i,v in enumerate(askdf.ix['AskPrice']):
        if( v == priceToDelete ):  
            askdf.pop( str(i+1) )
            depthAsk -=1
            askdf = askdf.ix[:,0:depthAsk]
            askdf.columns = map(str, np.arange(depthAsk)+1)
            break


# This whole thing takes 1209 seconds.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse CSVs,re-create order books and output HDFs')
    parser.add_argument('-i','--inputPath', help='Path of the raw .CSV files.',dest='in', required=True)
    parser.add_argument('-f','--fileName', help='Path of the raw .CSV files.',dest='file', required=True)
    parser.add_argument('-o','--outputPath', help='Path of the directory to store the HDF buckets.',dest='out', required=True)
    args = vars(parser.parse_args())    

    date= args['file'][-14:-4] # this is the date of every pair we are examining.
    if not os.path.exists(args['out']):
        os.makedirs(args['out'])
    f= open(args['in']+args['file'])
    output = open(args['out']+"fixed"+args['file'],'a')
    csvWriter = csv.writer(output)
    firstLine = True
    restartFlag = False
    
    start_time = time.time()
    while 1:
        line = f.readline()
        if line =="": break
        if( line.startswith('Q')):
            continue
        
        s = StringIO() # string buffer to read the messages and determine if a RESTART has occured.
        
        if(firstLine == True): #The header is a specific case where we read a line.
            assert(line.startswith("V3")) # File must be in version V.0.0.3
            output.write(line)
            firstLine = False
            line = f.readline() #move on to the next line
        
        while not (line.startswith('OB') or line.startswith('Q') ):
            if "RESTART" in line:
                restartFlag = True
                output.write(line) # write the line as we are supposed to.
                break
            output.write(line)
            s.write(line)                    
            line = f.readline()
            
        if line.startswith('OB'):
            s.write(line)
        s.seek(0)
        
        while 1:
            if(restartFlag): #if there is a restart message we don't alter the orderbook and not process any messages that we read in the buffer.
                restartFlag = False
                break
            
            line = s.readline() #otherwise we read each line of the buffer (which is the normal message from the exchange).
            if line=="": break    
        
            if ( line.startswith('A') or line.startswith('D') or line.startswith('OB')):
                row = line.split(',') 
                       
                if( row[0] == 'A'):
                    if( row[1] == '0'): #its a bid message.
                        bidAddition(float(row[3]), float(row[2]) )
                    elif(row[1] == '1'): #its an ask message.
                        askAddition(float(row[3]), float(row[2]) )
                    else:
                        print "Must Not Happen!"
                elif( row[0] == 'D'):
                    if( row[1] == '0'): #its a bid message.
                        bidDeletion(float(row[3]), float(row[2]))
                    elif(row[1] == '1'): #its an ask message.
                        askDeletion(float(row[3]), float(row[2]) )
                    else:
                        print "Must Not Happen!"            
                elif( row[0] == 'OB'):
                    finTS = None # Some TimeStamps have less digits than 9, so in this variable we save the correct representation of them.
                    #script to check for missing 0 in the nanoseconds position of the timestamp.
                    tsList = row[-1][:-1].split(':')
                    if len(tsList[1]) != 9:
                        tsList[1] = tsList[1].rjust(9,'0')
                    elif len(tsList[1]) > 9:
                        print "Number of nanoseconds is greater than 9. :-S "
                        exit()
                    # end of checking for missing zeros in the beginning.    
                    finTS = ":".join(tsList)
                    csvWriter.writerow(["OB",row[1],row[2], finTS] ) # Write the orderbook's header.
                    # then the 10 lines that are supposed to be in the file
                    for i in range(biddf.count(1)[0]):
                        csvWriter.writerow(["Q",row[2],'0',i+1, "%f"%biddf.ix[1,i] , int(biddf.ix[0,i]), int(biddf.ix[0,i]) ])
                    for i in range(askdf.count(1)[0]):
                        csvWriter.writerow(["Q",row[2],'1',i+1, "%f"%askdf.ix[1,i] , int(askdf.ix[0,i]), int(askdf.ix[0,i]) ])
                
                else: #just in case :)
                    print "Unknown Input!"
                    exit()
        
    print time.time() - start_time, "seconds" 