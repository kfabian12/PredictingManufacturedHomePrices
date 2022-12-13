#NaiveBayes.py
"""Predicting housing prices using Naive Bayes model."""

import argparse
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

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
    # Convert values in data to indices for regions
    training_data[:, 0] = training_data[:, 0] - 1
    testing_data[:, 0] = testing_data[:, 0] - 1

    # Make a list of possible values for shipdates
    shipdates = np.unique(training_data[:, 1])

    # Convert values in data to indices for shipdates
    training_data[:, 1] = ((training_data[:, 1] % 100) + 12 * np.floor(((training_data[:, 1] - 201401) / 100)) - 1).astype(int)
    testing_data[:, 1] = ((testing_data[:, 1] % 100) + 12 * np.floor(((testing_data[:, 1] - 201401) / 100)) - 1).astype(int)

    # Make a list of possible values for sqft
    sqft_categories = np.unique(training_data[:, 2])
    sqft_categories = np.unique(sqft_categories - (sqft_categories % 100))
    sqft_categories = np.delete(sqft_categories, 22)

    # Convert values in data to indices for sqft
    training_data[:, 2] = (training_data[:, 2] / 100) - 4
    training_data[training_data[:, 2] > 21, 2] = 21
    training_data[training_data[:, 2] < 0, 2] = 0

    testing_data[:, 2] = (testing_data[:, 2] / 100) - 4
    testing_data[testing_data[:, 2] > 21, 2] = 21
    testing_data[testing_data[:, 2] < 0, 2] = 0

    # Convert values in data to indices for bedrooms
    training_data[:, 3] = training_data[:, 3] - 1
    training_data[np.where(training_data[:, 3] == 2), 3] =  training_data[np.where(training_data[:, 3] == 2), 3] - 1

    testing_data[:, 3] = testing_data[:, 3] - 1
    testing_data[np.where(testing_data[:, 3] == 2), 3] =  testing_data[np.where(testing_data[:, 3] == 2), 3] - 1

    # Make a list of possible values for price
    possible_prices = np.unique(training_labels)
    possible_prices = np.unique(possible_prices - (possible_prices % 25000))
    
    # Convert values in labels to indices for price
    training_labels = np.floor(training_labels / 25000).astype(int)

    testing_labels = np.floor(testing_labels / 25000).astype(int)

    # A couple more variables
    num_train = len(training_labels)
    num_test = len(testing_labels)
    num_sqft = len(sqft_categories)
    num_prices = len(possible_prices)
    num_bedrooms = len(np.unique(training_data[:, 3]))
    num_regions = len(np.unique(training_data[:, 0]))
    num_shipdates = len(np.unique(training_data[:, 1]))
    num_parameters = num_regions + num_shipdates + num_sqft + num_bedrooms



    print("\n=======================")
    print("TRAINING")
    print("=======================")

    # Estimate the prior probabilities
    print("Estimating prior probabilities...")
    priors = np.bincount(training_labels) / num_train

    # Estimate the class conditional probabilities
    print("Estimating class conditional probabilities...")
    class_conditionals = np.zeros((num_parameters, num_prices))

    for i in range(num_prices):
        x = np.bincount(training_data[np.where(training_labels == i), 0].flatten()) 
        class_conditionals[:len(x), i] = x
        
        x = np.bincount(training_data[np.where(training_labels == i), 1].flatten()) 
        class_conditionals[num_regions:num_regions + len(x), i] = x

        x = np.bincount(training_data[np.where(training_labels == i), 2].flatten()) 
        class_conditionals[num_regions + num_shipdates:num_regions + num_shipdates + len(x), i] = x

        x = np.bincount(training_data[np.where(training_labels == i), 3].flatten()) 
        class_conditionals[num_parameters - num_bedrooms:num_parameters - num_bedrooms + len(x), i] = x

    alpha = 1/num_train
    class_conditionals += alpha
    class_conditionals /= np.sum(class_conditionals, 0)
    
    pdb.set_trace()



    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)
    log_class_conditionals = np.log(class_conditionals)

    print("Computing parameters in data...")
    counts = np.zeros((num_test, num_parameters))

    for i in range(num_prices):
        pass

    print("Computing posterior probabilities...")
    pdb.set_trace
    log_posterior = np.matmul(counts, log_class_conditionals)
    log_posterior += log_priors

    print("Assigning predictions via argmax...")
    pred = np.argmax(log_posterior, 1)

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics
    accuracy = np.mean(testing_labels == pred)
    print("Accuracy: {0:d}/{1:d} ({2:0.1f}%)".format(sum(testing_labels == pred), num_test, accuracy * 100))
    cm = confusion_matrix(testing_labels, pred)
    print("Confusion matrix:")
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))


if __name__ == "__main__":
    main()