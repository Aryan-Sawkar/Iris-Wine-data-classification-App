import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title('Iris & Wine Classification Machine Learning App')
st.subheader("Exlplore Different Classifiers")
st.text("which one is the best")

dataset_name=st.sidebar.selectbox("Select Dataset",("Iris",'Wine'))
st.write(dataset_name)

classifier_name=st.sidebar.selectbox("Select classifier",("KNN",'SVM'))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)
st.write('Shape of the Dataset',X.shape)
st.write('Number of classes',len(np.unique(y)))


def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    else:
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    return params

params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
       clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=SVC(C=params["C"])

    return clf
clf=get_classifier(classifier_name,params)

#Classification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"Accuracy={acc}")

#Plot
pca=PCA(2)
X_projected=pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
