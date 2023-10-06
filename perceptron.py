import numpy as np

class Perceptron:
    """
    N: The number of columns in our input feature vectors.
       In the context of our bitwise datasets, we’ll set N equal to two since there are two inputs.
    alpha: Our learning rate for the Perceptron algorithm. We’ll set this value to 0.1 by default.
           Common choices of learning rates are normally in the range α = 0.1, 0.01, 0.001.
    """
    def __init__(self,N,alpha=0.1) -> None:
        self.w = np.ones(shape=(1, N))
        self.alpha = alpha
    

    def step(self,x):

        return 1 if x > 0 else 0
    
    def fit(self,X,y,epochs=10):
        misclassified_ = [] 
        for epoch in np.arange(0, epochs):
            misclassified = 0
			# loop over each individual data point
            for (x, target) in zip(X, y):
          
				# take the dot product between the input features
				# and the weight matrix, then pass this value
				# through the step function to obtain the prediction
                #x = np.insert(x,0,1)
                output = np.dot(x,self.w.T).item()
                p = self.step(output)
				# only perform a weight update if our prediction
				# does not match the target
				# determine the error
                error = p-target.item()
                misclassified += error
                # update the weight matrix
                self.w += -self.alpha * x * error

            misclassified_.append(misclassified)
        return (self.w, misclassified_)

    def predict(self,X):
        X = np.atleast_2d(X)
        return self.step(np.dot(X,self.w.T))