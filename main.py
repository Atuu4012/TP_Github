import pandas as pd 
from sklearn import svm  
from train_model import train_model
from preprocess_data import preprocess_data

iris = pd.read_csv("InputData/Iris.csv") #load the dataset
test_size = 0.3 # the attribute test_size=0.3 to use for splitting the data 
				#into 70% for train and 30% for test

train, test =preprocess_data(iris, test_size)
# training data features
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# target of our training data
train_y=train.Species
# test data features
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
#target value of test data
test_y =test.Species   


model = svm.SVC()
prediction = train_model(train_X, train_y, test_X, model)


