from perceptron import Perceptron
import numpy as np


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])
y = np.array([

    [0],
    [1],
    [1],
    [1],
])

print("Training Perceptron OR...")
p = Perceptron(X.shape[1],alpha=0.1)
p.fit(X,y,epochs = 1)

print(f" trained weight : {p.w}")

# now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")
# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))