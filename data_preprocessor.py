# Data Preprocessor
#
# Note you need to change the name of the file that you read in from
# at line 43 (Currently Set to the Kindle Store example)
#
#
# Critical Value is the minimum number of reviews which a book must have
# to be worth analyzing. It can be changed at line
#
# Written by Mark Mori


import pandas
import numpy

from pathlib import Path
from warnings import simplefilter
simplefilter(action="ignore", category=pandas.errors.PerformanceWarning)
import multiprocessing
from functools import partial



def make_csv(elem,histories):

    fpath=Path('data3/'+elem['asin'].unique()[0]+'.csv')
    #elem=elem.reindex(new_col_list, axis='columns')
    elem.reset_index(inplace=True,drop=True)

    #Significance threshhold
    for i in range(len(elem)):
        for review in histories[elem.at[i,'reviewerID']]:
            if review!=elem.at[i,'asin']:
                elem.at[i,review]=histories[elem.at[i,'reviewerID']][review]

    elem.drop(['asin','reviewerID'],inplace=True,axis='columns')
    elem.to_csv(fpath,index=False)
            #elem no longer necessary. Don't take up room in memory

if __name__ == '__main__':

    dataset="Gift_Cards_5"
    Critical=10

    print("Starting",flush=True)

    df = pandas.read_json(dataset+".json",lines=True)

    print("Finished Reading",flush=True)

    df.drop(['reviewTime','unixReviewTime','image','style','reviewText','summary','verified','reviewerName'],axis=1,inplace=True)
    df['vote']=pandas.to_numeric(df['vote'].replace(',',''), errors='coerce')
    df['vote']=df['vote'].replace(numpy.nan,0)

    print("---Data Statistics---")
    print("# Total Reviews: \t",len(df['overall']))
    print("# Unique Reviewers:\t", len(df['reviewerID'].unique()))
    print("# Unique Items: \t", len(df['asin'].unique()),"\n")

    print("Creating Reviewer Histories.",flush=True)
    list_histories={}
    for i,entry in enumerate(df.itertuples(),1):
        if entry.reviewerID not in list_histories:
            list_histories[entry.reviewerID]={}
        list_histories[entry.reviewerID][entry.asin]=entry.overall

    print("Creating CSVs...",flush=True)

    old_col_list=df.columns.tolist()
    new_col_list=df['asin'].unique().tolist()+old_col_list

    #Master CSV with only the information relevant for training
    master=df.copy()
    master.drop(['reviewerID','vote'],axis='columns',inplace=True)
    master.to_csv(dataset+"_master.csv",index=False)

    df = df.groupby('asin')
    dataframes = [group for _, group in df]
    Critical_dataframes=[]
    for item in dataframes:
        if len(item)>=Critical:
            Critical_dataframes.append(item)

    tasks_pool=multiprocessing.Pool()
    make=partial(make_csv,histories=list_histories)

    finished=tasks_pool.map_async(make,Critical_dataframes)

    finished.wait()