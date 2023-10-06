import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron

def loadData():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)
    print(data)
    
    # make the dataset linearly separable
    data = data[:]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, np.where(data.iloc[:, -1] == 'Iris-versicolor', 1, 2))
    data = np.asmatrix(data, dtype = 'float64')
    return data
data = loadData()


plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:100,0]), np.array(data[50:100,2]), marker='x', label='versicolor')
plt.scatter(np.array(data[100:,0]), np.array(data[100:,2]), marker='*', label='virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()

np.random.shuffle(data)



features        = data[20:, :-1]
target          = data[20:, -1]

testFeatures    = data[:20, :-1]
testTarget      = data[:20, -1]

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