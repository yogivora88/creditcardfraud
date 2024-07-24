import pandas as pd
from sklearn.datasets import make_classification
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

Data = pd.read_csv(r"C:\Users\yoges\OneDrive\Desktop\creditcard.csv")

Data

Data.columns



x_Data = Data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']]
y_Data = Data["Amount"]

from sklearn.preprocessing import MinMaxScaler

x_Data.head()

scalar = MinMaxScaler()
scalar.fit(x_Data)
New_Data = scalar.transform(x_Data)

New_Data

Data.shape

!pip install imbalanced-learn

x, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=[0.05, 0.1, 0.85],
                           class_sep = 0.8, random_state=0)

x[:,0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
sns.scatterplot(x=x[:,0], y=x[:,1], hue=y, ax=ax[0])
idx, c = np.unique(y, return_counts=True)
sns.barplot(x=idx, y=c, ax=ax[1])
plt.show()

Data.outcome.value_counts()

x = Data.iloc[:,:-1]
y = Data.Time
x.head()

pip install imblearn

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = \
train_test_split(x,y,test_size=0.3,random_state=10)

pip install scikit-learn

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

from sklearn.metrics import accurancy_score
print(accuracy_score(y_test,y_predict))

