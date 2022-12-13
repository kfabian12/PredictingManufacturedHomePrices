<<<<<<< HEAD
# Predict Manufactured Home Prices
# K-Nearest Neighbor

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib.colors import ListedColormap
import pdb


ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    
    cmap_light = ListedColormap(['#FFAFAF', '#AFAFFF', '#F6D587'])
    cmap_bold = ListedColormap(['red', 'blue', 'orange'])
    
    # import data
    print("Loading training data...")
    training = np.loadtxt(os.path.join(ROOT, 'trainingdata.csv'), delimiter=',', dtype=int)
    x_train = training[:, :-1]
    y_train = training[:, -1]
    print("Loading testing data...")
    testing = np.loadtxt(os.path.join(ROOT, 'testingdata.csv'), delimiter=',', dtype=int)
    x_test = testing[:, :-1]
    y_test = testing[:, -1]
    
    x_train[:, 2] = (x_train[:, 2] / 100) - 4
    x_train[x_train[:, 2] > 21, 2] = 21
    x_train[x_train[:, 2] < 0, 2] = 0

    x_test[:, 2] = (x_test[:, 2] / 100) - 4
    x_test[x_test[:, 2] > 21, 2] = 21
    x_test[x_test[:, 2] < 0, 2] = 0

    # make a list of all possible values for price
    possible_prices = np.unique(y_train)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    # convert values to indices
    y_train = np.floor(y_train / 25000).astype(int)
    y_test = np.floor(y_test / 25000).astype(int)
    
    # train k-NN
    knn = KNeighborsRegressor(n_neighbors=8)
    knn.fit(x_train, y_train)
  
    # Predict on the test data
    pred = knn.predict(x_test)
    pred = pred.astype(int)
    print(pred)
    
    # Calculate the accuracy of the model using test data
    print("Calculating accuracy...")
    print(knn.score(x_test, y_test))
    
    # show results
    cm = confusion_matrix(y_test, pred)
    print(cm)
    
    xm, ym = np.meshgrid(np.arange(-0.1, 1.1, 0.002), np.arange(-0.1, 1.1, 0.002))
    plt.pcolormesh(xm, ym, pred, shading='auto')
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cmap_bold, edgecolor='k', s=100)
    plt.title("Predict Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('SQFT')
    plt.show()
        
    pdb.set_trace()
    
    

if __name__ == "__main__":
    main()
=======
# Predict Manufactured Home Prices
# K-Nearest Neighbor

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib.colors import ListedColormap
import pdb


ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    
    cmap_light = ListedColormap(['#FFAFAF', '#AFAFFF', '#F6D587'])
    cmap_bold = ListedColormap(['red', 'blue', 'orange'])
    
    # import data
    print("Loading training data...")
    training = np.loadtxt(os.path.join(ROOT, 'trainingdata.csv'), delimiter=',', dtype=int)
    x_train = training[:, :-1]
    y_train = training[:, -1]
    print("Loading testing data...")
    testing = np.loadtxt(os.path.join(ROOT, 'testingdata.csv'), delimiter=',', dtype=int)
    x_test = testing[:, :-1]
    y_test = testing[:, -1]
    
    x_train[:, 2] = (x_train[:, 2] / 100) - 4
    x_train[x_train[:, 2] > 21, 2] = 21
    x_train[x_train[:, 2] < 0, 2] = 0

    x_test[:, 2] = (x_test[:, 2] / 100) - 4
    x_test[x_test[:, 2] > 21, 2] = 21
    x_test[x_test[:, 2] < 0, 2] = 0

    # make a list of all possible values for price
    possible_prices = np.unique(y_train)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    # convert values to indices
    y_train = np.floor(y_train / 25000).astype(int)
    y_test = np.floor(y_test / 25000).astype(int)
    
    # train k-NN
    knn = KNeighborsRegressor(n_neighbors=8)
    knn.fit(x_train, y_train)
  
    # Predict on the test data
    pred = knn.predict(x_test)
    pred = pred.astype(int)
    print(pred)
    
    # Calculate the accuracy of the model using test data
    print("Calculating accuracy...")
    print(knn.score(x_test, y_test))
    
    # show results
    cm = confusion_matrix(y_test, pred)
    print(cm)
    
    xm, ym = np.meshgrid(np.arange(-0.1, 1.1, 0.002), np.arange(-0.1, 1.1, 0.002))
    plt.pcolormesh(xm, ym, pred, shading='auto')
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cmap_bold, edgecolor='k', s=100)
    plt.title("Predict Manufactured Housing Prices")
    plt.xlabel('Year')
    plt.ylabel('SQFT')
    plt.show()
        
    pdb.set_trace()
    
    

if __name__ == "__main__":
    main()
>>>>>>> 161dd78e0adafe00038e6661cb9f424cc2cbf7bf
