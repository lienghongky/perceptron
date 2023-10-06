import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron

def loadData():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)
    print(data)
    
    # make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data
data = loadData()



print(data[:10,:-1].shape)
print(data[50:60,:-1].shape)
features        = np.concatenate((data[10:50, :-1] , data[60:,:-1]),axis=0)
target          = np.concatenate((data[10:50, -1] , data[60:,-1]),axis=0)

testFeatures    = np.concatenate((data[:10, :-1]    , data[50:60,:-1]),axis=0)
testTarget      = np.concatenate((data[:10, -1]    , data[50:60,-1]),axis=0)

print(f"Training on :{features.shape}...")
p = Perceptron(features.shape[1],alpha=0.1)
nEpoch = 100000
w,misclassified_ = p.fit(features,target,epochs=nEpoch)

print(f"Testing on :{testFeatures.shape}... \n W = {p.w}")
miss = 0
for (x, target) in zip(testFeatures, testTarget):
	# make a prediction on the data point and display the result
	# to our console
    print(x)
    pred = p.predict(x)
    if pred != target[0]:
        miss += 1
    print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
print(f"RESULT: loss - {miss/testFeatures.shape[0]}")

epochs = np.arange(1, nEpoch+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()