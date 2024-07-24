!pip install pandas

import pandas as pd

dataframe = pd.read_csv(r"C:\Users\yoges\OneDrive\Desktop\creditcard.csv")

print("our dataframe....",dataframe)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv(r"C:\Users\yoges\OneDrive\Desktop\creditcard.csv")

dataframe.head()

dataframe.tail()

dataframe.shape

dataframe.info()

dataframe.isnull().sum()

dataframe.duplicated().sum()

dataframe.describe().T

dataframe.describe(include="float64")

for i in dataframe.select_dtypes(include="number").columns:
    sns.histplot(data=dataframe,x=i)
    plt.show()

for i in dataframe.select_dtypes(include="number").columns:
    sns.boxplot(data=dataframe,x=i)
    plt.show()

for i in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'] :
    sns.scatterplot(data=dataframe,x=i,y='Time')
    plt.show()

dataframe.select_dtypes(include="number").columns

s=dataframe.select_dtypes(include="number").corr()

plt.figure(figsize=(30,60))
sns.heatmap(s,annot=True)



def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw

wisker(dataframe['V14'])

for i in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']:
    lw,uw=wisker(dataframe[i])
    dataframe[i]=np.where(dataframe[i]<lw,lw,dataframe[i])
    dataframe[i]=np.where(dataframe[i]<uw,uw,dataframe[i])

for i in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']:
    sns.boxplot(dataframe[i])
    plt.show

dataframe.columns

dataframe.drop_duplicates()







