# Master Trainer
# 
# Just like with data_preprocessor
# The dataset and critical value
# must me defined (Line 45-46)
#
# Written by Mark Mori

import matplotlib.pyplot as plt
import pandas
import xgboost
import numpy
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import train_test_split

from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

import multiprocessing


def train_test_report(asin):
    
    file=Path("data/"+asin+".csv")
    df = pandas.read_csv(file)
    
    model=xgboost.XGBRegressor(objective='reg:squarederror',eval_metric='rmse',early_stopping_rounds=20,seed=1,max_depth=0,learning_rate=.1,gamma=.1,reg_lambda=1,device='cpu',subsample=.8)
    #model=xgboost.XGBClassifier(objective='multi:softmax',eval_metric='auc',early_stopping_rounds=20,seed=1,max_depth=0,learning_rate=.1,gamma=.1,reg_lambda=1,device='cpu',subsample=.8)

    x = df.drop(['overall'],axis=1).copy()
    y = df['overall'].copy() 
    
    #For classification
    #y = y.add(-1)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)
    wieght_table = compute_sample_weight('balanced', y_train)

    model.fit(x_train,y_train,verbose=False,sample_weight=wieght_table,eval_set=[(x_test,y_test)])

    y_predicted=model.predict(x_test)

    return (y_test,y_predicted)

if __name__ == '__main__':

    Dataset="Gift_Card_5"
    Critical=10

    print("Reading Master",flush=True)
    df = pandas.read_csv(Dataset+"_master.csv")
    print("Finding elements Such that the critical minimum is met. Critical = ",Critical,flush=True)

    df=df.groupby('asin')
    df_list = [group for _, group in df]
    df=pandas.DataFrame() #don't take up space in memory

    list_critical=[]

    for item in df_list:
        if len(item['asin'])>=Critical:
            list_critical.append(item.iat[0,1])
    

    print("Beginning Training",flush=True)


    training_pool=multiprocessing.Pool()

    results=training_pool.map_async(train_test_report,list_critical)

    results.wait()

    print("Finding Statistics",flush=True)

    list_true=[]
    list_predicted=[]

    for result in results.get():
        if result!=False:   
            list_true.extend(result[0])
            list_predicted.extend(result[1])
        else:
            print(False)

    #Rounding for Regression
    list_predicted=[round(x) for x in list_predicted] 
    list_predicted=[min(x,5)for x in list_predicted]
    list_predicted=[max(x,1)for x in list_predicted]

    #For Classification
    #list_predicted=[x+1 for x in list_predicted]
    #list_true=[x+1 for x in list_true]
    #map(lambda x:int(round(x,0)),list_true)

    true_pos_dict=dict.fromkeys(range(1,6),0)
    false_pos_dict=dict.fromkeys(range(1,6),0)
    false_neg_dict=dict.fromkeys(range(1,6),0)

    mean_absolute_error=0.0
    for predict,truth in zip(list_predicted,list_true):
            
        if (predict==truth): true_pos_dict[truth]+=1
        if (predict!=truth): false_pos_dict[predict]+=1
        if (predict!=truth): false_neg_dict[truth]+=1
        mean_absolute_error+=abs(truth-predict)

    mean_absolute_error=mean_absolute_error/len(list_true)
    root_mean_absolute_error=mean_absolute_error**.5

    for x in range(1,6):
            print("Precision\t",x," = ",true_pos_dict[x]/(true_pos_dict[x]+false_pos_dict[x]))
            print("Recall \t\t",x," = ",true_pos_dict[x]/(true_pos_dict[x]+false_neg_dict[x]))

    print("Accuracy:\t",sum([true_pos_dict[x] for x in range(1,6)])/len(list_predicted))
    print("MAE: \t",mean_absolute_error)
    print("RMSE:\t",root_mean_absolute_error)

    ConfusionMatrixDisplay.from_predictions(y_true=numpy.array(list_true),y_pred=numpy.array(list_predicted))
    plt.savefig("out.png")