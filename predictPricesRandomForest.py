# Predict Manufactured Home Prices
# Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix  
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Predicting Housing Prices using Naïve Bayes Classifier")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Load data from relevant files
    print("Loading training data...")
    training = np.loadtxt(os.path.join(ROOT, 'trainingdata.csv'), delimiter=',', dtype=int)
    training_data = training[:, :-1]
    training_labels = training[:, -1]
    print("Loading testing data...")
    testing = np.loadtxt(os.path.join(ROOT, 'testingdata.csv'), delimiter=',', dtype=int)
    testing_data = testing[:, :-1]
    testing_labels = testing[:, -1]

    # Extract useful parameters
    # Make a list of possible values for sqft
    sqft_categories = np.unique(training_data[:, 2])
    sqft_categories = np.unique(sqft_categories - (sqft_categories % 100))
    sqft_categories = np.delete(sqft_categories, 22)

    #convert values to indices
    training_data[:, 2] = (training_data[:, 2] / 100) - 4
    training_data[training_data[:, 2] > 21, 2] = 21
    training_data[training_data[:, 2] < 0, 2] = 0

    testing_data[:, 2] = (testing_data[:, 2] / 100) - 4
    testing_data[testing_data[:, 2] > 21, 2] = 21
    testing_data[testing_data[:, 2] < 0, 2] = 0

    #Make a list of possible values for price
    possible_prices = np.unique(training_labels)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    #convert values to indices
    training_labels = np.floor(training_labels / 25000).astype(int)

    testing_labels = np.floor(testing_labels / 25000).astype(int)

    
    # create classifier object  
    classifier= RandomForestRegressor(n_estimators= 10, criterion="entropy")  
    classifier.fit(x_train, y_train)  
    
    
    
    # predict the test result
    y_pred= classifier.predict(x_test)  
    
    # create confusion matrix to determine correct predictions
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    pdb.set_trace()
    
    # X_grid = np.arrange(min(x_train), max(x_train), 0.01)                   
    # X_grid = X_grid.reshape((len(X_grid), 1))
    # plt.scatter(x_train, y_train, color = 'blue')  
    # plt.plot(X_grid, y_pred, color = 'green') 
    # plt.title('Predict Manufactured Housing Prices')
    # plt.xlabel('Square Footage')
    # plt.ylabel('Price')
    # plt.show()
    

if __name__ == "__main__":
    main()

