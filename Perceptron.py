import pandas as pd
import math
import random
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Perceptron:
    def fit(self, X, y, learning_rate=0.5, iterations=100000):
        D=X.shape[1]
        self.w=np.random.randn(D)
        self.b=0.5
        N=len(y)
        costs=[]
        for epoch in range(iterations):
            yhat=self.predict(X)
            incorrect=np.nonzero(y!=yhat)[0]
            if len(incorrect)==0:
                break
            i=np.random.choice(incorrect)
            self.w+=learning_rate*y[i]*X[i]
            self.b+=learning_rate*y[i]
            c=len(incorrect)/float(N)
            costs.append(c)
        print ("final w:", self.w, "final b:", self.b, "epochs:", (epoch+1) ,"/", iterations)
        plt.plot(costs)
        plt.show()


    def predict(self, X):
      return np.sign(X.dot(self.w)+self.b)

    def score(self, X, y):
        p=self.predict(X)
        return np.mean(p==y)

    def test(self , INPUTS , OUTPUTS):
        accuracy = 0
        for i in range(len(OUTPUTS)):
            if(self.predict(INPUTS[i]) == OUTPUTS[i]):
                accuracy += 1
        return accuracy / (float)(len(OUTPUTS))






def main():

    dataset =pd.read_csv('wpbc.data') #pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data')

    start = 3
    X = dataset.iloc[:, start:].values
    w = np.random.randn(X.shape[1])
    y = dataset.iloc[:, 1].values  # this is the train output array. N or R
    le =LabelEncoder()# transform output from N and R to 0 and 1
    y=le.fit_transform(y)
    #  spliting the data set to 80% train, 20% test.
    x_train, x_test, y_train, y_test= train_test_split(X,y ,test_size=0.38, random_state=1)
    p = Perceptron()
    start = timeit.default_timer()
    p.fit( x_train,  y_train,0.01,1000)

    print("score", p.score(x_test, y_test)*100, "%")
    stop = timeit.default_timer()
    print('Time ', stop - start)

    print('Accuracy ' , p.test(x_test , y_test) * 100 , '%')




if __name__ == '__main__':
    main()