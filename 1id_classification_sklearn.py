

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

Categories=['HEDM','Powder'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='1ID_data/'
#path which contains all the categories of images 
for i in Categories: 
	
	print(f'loading... category : {i}') 
	path=os.path.join(datadir,i) 
	for img in os.listdir(path): 
		img_array=imread(os.path.join(path,img)) 
		img_resized=resize(img_array,(224,224)) 
		flat_data_arr.append(img_resized.flatten()) 
		target_arr.append(Categories.index(i)) 
	print(f'loaded category:{i} successfully') 
flat_data=np.array(flat_data_arr) 
target=np.array(target_arr)

#dataframe 
df=pd.DataFrame(flat_data) 
df['Target']=target 
df.shape

#input data 
x=df.iloc[:,:-1] 
#output data 
y=df.iloc[:,-1]

# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, 
											random_state=77, 
											stratify=y) 


model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))

confusion_matrix(y_pred,y_test)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)

accuracy_score(y_pred_knn,y_test)
print(classification_report(y_pred_knn,y_test))
confusion_matrix(y_pred_knn,y_test)


dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc=dtc.predict(x_test)

accuracy_score(y_pred_dtc,y_test)
print(classification_report(y_pred_dtc,y_test))
confusion_matrix(y_pred_dtc,y_test)


path='1ID_data/test.jpg'
img=imread(path) 
plt.imshow(img) 
plt.show() 
img_resize=resize(img,(224,224)) 
l=[img_resize.flatten()] 
probability=model.predict_proba(l) 
for ind,val in enumerate(Categories): 
	print(f'{val} = {probability[0][ind]*100}%') 
print("The predicted image is : "+Categories[model.predict(l)[0]])

