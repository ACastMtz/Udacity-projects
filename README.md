# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

In this project, the UCI Bank Marketing dataset was used to train a model to predict if a client is likely to subscribe to a term deposit with the bank.

To that end, the parameters of a Scikit-learn logistic regression model were automatically optimized using the HyperDrive package from the Azure Machine Learning studio with the Python SDK.

Additionally, an AutoML pipeline was used to train and tune a model on the same dataset to afterwards compare the results from both methods.

In both cases the target metric was the model's "Accuracy". 
The Hyperdrive run yielded an accuracy of 0.9073 for a logistic regression model and from the AutoML run the highest reached accuracy was 0.9178 using a VotingEnsemble method.

## Scikit-learn Pipeline

CREATE AZURE MACHINE LEARNING COMPUTE CLUSTER:
Create the the compute cluster in which the Hyperdrive run will execute.
The compute is composed by 4 two-core nodes, which will start by provisioning 1 node and scale up to 4 as required (vm Standard_D2_V2).

DATA PREPARATION, CONFIGURATION AND MODEL TRAINING:
First the data is downloaded as a TabularDataset to begin its preparation. 
It is then cleaned by transforming cathegorical values into numerical ones via the get_dummies function and applying lambda functions to perform one hot encoding. Afterwards, it is split in training and test datasets. The dataset are then fed into the main function where the logistic regression model is trained, fitted and scored (using as metric the accuracy of the model). The initial hyperparameter values are parsed arguments with default values: inverse of regularization strength "C" = 1.0 and maximum number of iterations "max_iter" = 1000. These three values, C, max_iter and accuracy, are also logged.
The steps described above are coded in the script "train.py".

HYPERPARAMETER TUNING:
A Hyperdrive run is set up to allow for automatically sweeping and finding the best values for the adjustable parameters of the custom-coded logistic regression model:

1st define parameter search space: continuous for C and discrete for max_iter.
2nd define sampling method,i.e. how do you want to find the best values over the search space. In this case a random sampling approach was used due to its time efficiency and still yielding good results.
3rd specify an eraly termination policy to stop the model's tuning after a certain number of failures.
4th create a Sklearn estimator 
5th specify the primary metric to optimize, in our case accuracy.



Finally, use the mode parameters found to tune the final machine learning model.

A grid sampling is exhaustive, but more time-consuming. In contrast, a random sweep can get good results without taking as much time. (In most of the cases preferable when time efficiency is more important because the "improvement" through grid search is not significant.)
maybe Bayesian sampling better (selects values based on how previous values improved the training performance; some child runs showed the same accuracy at different parameter values so the whole sampling needn't be carried out completely.)



## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
validation set for validation of hyperparameters
data is unbalanced --> look at better metrics, data featurization, data engineering
choose better model for hyperdrive
longer run time
grid search sampling, bayesian
use gpu compute and enable deep learning models

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
