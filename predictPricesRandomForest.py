# Predict Manufactured Home Prices
# Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def main():
    # import data
    x_train = pd.read_csv("trainingdata.csv", usecols=[0,1,2,3], header=None).values
    y_train = pd.read_csv("trainingdata.csv", usecols=[4], header=None).values
    x_test = pd.read_csv("testingdata.csv", usecols=[0,1,2,3], header=None).values
    y_test = pd.read_csv("testingdata.csv", usecols=[4], header=None).values
    
    # create classifier object  
    classifier= RandomForestRegressor(n_estimators= 10, criterion="entropy")  
    classifier.fit(x_train, y_train)  
    
    pdb.set_trace()
    
    # predict the test result
    y_pred= classifier.predict(x_test)  
    
    # create confusion matrix to determine correct predictions
    cm = confusion_matrix(y_test, y_pred)
    
    X_grid = np.arrange(min(x_train), max(x_train), 0.01)                   
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x_train, y_train, color = 'blue')  
    plt.plot(X_grid, y_pred, color = 'green') 
    plt.title('Predict Manufactured Housing Prices')
    plt.xlabel('Square Footage')
    plt.ylabel('Price')
    plt.show()
    

if __name__ == "__main__":
    main()
