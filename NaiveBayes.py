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
    


    pdb.set_trace()


    print("\n=======================")
    print("TESTING")
    print("=======================")

    # Test the Naive Bayes classifier
    print("Applying natural log to prevent underflow...")
    log_priors = np.log(priors)

    print("Counting words in each document...")

    print("Computing posterior probabilities...")

    print("Assigning predictions via argmax...")


    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compute performance metrics



if __name__ == "__main__":
    main()