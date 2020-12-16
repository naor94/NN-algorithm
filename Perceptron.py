import pandas as pd
import math
import random
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class Perceptron(object):
    def __init__(self, eta=0.001, n_iter=500):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        self.w=np.zeros(1+X.shape[1])
        self.errors=[]
        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip (X,y):
                update=self.eta*(target-self.predict(xi))
                self.w[1:]+=update*xi
                self.w[0]+=update
                errors+=int(update!=0.0)
            self.errors.append(errors)
        print("final w:", self.w)
        return self
    def net_input(self, X):
        return np.dot(X, self.w[1:])+self.w[0]
    def predict(self, X):
         return np.where(self.net_input(X)>=0.0,1,-1)



    def test(self, INPUTS, OUTPUTS):
        accuracy = 0
        for i in range(len(OUTPUTS)):
            if (self.predict(INPUTS[i]) == OUTPUTS[i]):
                accuracy += 1
        return accuracy / (float)(len(OUTPUTS))




def main():

    df=pd.read_csv('wpbc.data', header=None)
    X = df.iloc[:, 3:].values
    y=df.iloc[:, 1].values
    y=np.where(y=="N",-1,0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    p = Perceptron()
    start = timeit.default_timer()
    p.fit(x_train,y_train)
    stop = timeit.default_timer()
    print('Time ', stop - start)
    print('Accuracy ' , p.test(x_test,y_test) * 100 , '%')
    plt.plot(range(1,len(p.errors)+1), p.errors)
    plt.xlabel('Attempts')
    plt.ylabel('Number of misclassification')
    plt.show()



if __name__ == '__main__':
    main()