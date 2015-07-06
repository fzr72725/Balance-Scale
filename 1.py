import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split
from sklearn import metrics

data = pd.read_csv('data.csv')
#print data.head()
#print pd.isnull(data).any().any()
data['Class_Num'] = 4
dict = {'L':0,'B':1,'R':2}
data['Class_Num'] = data['Class_Name'].map(dict).astype(int)
data['Left'] = 1
data['Left']=data['Left-Weight']*data['Left-Distance']
data['Right'] = 1
data['Right']=data['Right-Weight']*data['Right-Distance']
data['Center'] = 0
data['Center'] = data['Left']-data['Right']
#print data['Class_Num'].head()

data_attr = data[data['Center']!=0][['Left','Right']]
data_targ = data[data['Center']!=0]['Class_Num']
#print data_attr.shape
X = data_attr
y = data_targ

#plt.scatter(X['Right-Weight']*X['Right-Distance']-X['Left-Weight']*X['Left-Distance'],y)
#plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)
