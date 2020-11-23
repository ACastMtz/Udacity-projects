# Optimizing an ML Pipeline in Azure

## Overview
\
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
\
In this project, the UCI Bank Marketing dataset was used to train a model to predict if a client is likely to subscribe to a term deposit with the bank.

To that end, the parameters of a **Scikit-learn logistic regression** model were automatically optimized using the **HyperDrive** package from the Azure Machine Learning studio with the Python SDK.

Additionally, an **AutoML** pipeline was used to train and tune a model on the same dataset to afterwards compare the results from both methods.

In both cases the target metric was the model's **Accuracy**. 
The Hyperdrive run yielded an accuracy of **0.9073 for a logistic regression model** and from the AutoML run the highest reached accuracy was **0.9178 using the VotingEnsemble method**.

## Scikit-learn Pipeline
\
CREATE AZURE MACHINE LEARNING COMPUTE CLUSTER:

Create the the compute cluster in which the Hyperdrive experiment will execute.
The compute is composed by 4 two-core nodes, which will start by provisioning 1 node and scale up to 4 as required (vm Standard_D2_V2).

DATA PREPARATION, CONFIGURATION AND MODEL TRAINING:

First the data is downloaded as a TabularDataset to begin its preparation. 
It is then cleaned by transforming cathegorical values into numerical ones via the get_dummies function and applying lambda functions to perform one hot encoding. Afterwards, it is split in training and test datasets. The datasets are then fed into the main function where the logistic regression model is trained (for classification), fitted and scored (using as metric the accuracy of the model). The initial hyperparameter values are parsed arguments with default values: **inverse of regularization strength "C" = 1.0** and **maximum number of iterations "max_iter" = 1000**. These three values, C, max_iter and accuracy, are also logged. Other important parameters are: *"penalty"=l2* (default) and *"solver" = lbfgs* (Limited-memory approximation of Broyden-Fletcher-Goldfarb-Shanno algorithm, default) to use in the optimization of the algorithm.
The steps described above are coded in the script **"train.py"**.

HYPERPARAMETER TUNING AND VALIDATION:

A Hyperdrive run is set up to allow for automatically sweeping and finding the best values for the adjustable parameters of the custom-coded logistic regression model:

1st--> define *parameter search space*: continuous for C and discrete for max_iter.

2nd--> define *sampling method*,i.e. how do you want to find the best values over the search space. In this case a random sampling approach was used due to its time efficiency while still yielding good results.

3rd--> specify an early *termination policy* to stop the model's tuning after a certain number of failures in order to improve computational efficiency. In this project, a **Bandit Policy** was used which terminates a specific run if its primary metric is not higher than a reference value called **Slack_factor** that defines the allowed "slack" of a single run's primary metric compared to the best performing run. The frequency at which the policy is applied is set by the **evaluation_interval**, if it is 1, the policy will be applied at every interval when metrics are reported. A Bandit Policy provides a better savings scheme (assuming a small enough slack) that may incur from compute resources by aggresively terminating runs.

4th--> create a *Sklearn estimator* to run the script-based training, which will be executed using the compute configuration that was defined at the beginning of the experiment. An Estimator object is helpful when the Run Configuration is too complex. The constructor takes in values such as the compute resources allocated for experiment execution and the training script. Particularly, the Sklearn estimator includes the framework's (SKLearn) specific dependencies.

5th--> specify the primary metric to optimize, in our case **Accuracy**.

6th--> create a *HyperDriveConfig* with the estimator, sampling method and termination policy. The **primary_metric_goal (maximize the accuracy)**, max_total_runs and max_concurrent_runs can be defined here.

Finally, the hyperdrive run is submitted and the model from the best run saved.


## AutoML
\
The AutoML classification task was run with similar parameters on the same dataset using **Accuracy** as the primary metric. As already mentioned, the **VotingEnsemble method** yielded the highest accuracy value. A voting ensemble is a machine learning model that builds predictions from combining results from other models, it is therefore sometimes refered to as a meta-model. More specifically, Azure AutoML uses the **PreFittedVotingClassifier Class** (inherited from sklearn *ensemble VotingClassifier* class) with the parameters following parameters:

  Estimators: Models to include in the voting classifier
  
  Weights: Weights for each estimator
  
  flatten_transform: Defines the way the results are displayed

## Pipeline comparison
\
As a first noticeable difference between the Hyperdrive and the AutoML runs is the higher accuracy provided by the latter method: **0.9073 Hyperdrive < 0.9178 AutoML**. One possible explanation for this is that with Hyperdrive only one model was trained, AutoMl on the other hand trained multiple models and then applied a voting classifier at the end. The advantage of the voting classifier is that it is very useful when the each of combined classifiers already show good performance and through combining them their indiviual weaknesses are reduced. 

Another difference is the time it took for each run to complete: **Hyperdrive appr. 10 min** and **AutoML appr. 20**. This is consistent due to AutoML's thourough search for a good model, it is expected that it takes longer to sweep multiple different models with different parameters, instead of just one custom-coded model as Hyperdrive does. However, one thing worth mentioning is that the fact that Hyperdrive uses a user-defined training script allows for a more direct control over the whole training process, making it easier to follow the pipeline and adjust if necessary to achieve the goal at hand; this could be making a highly accurate model or to expedite the training while maintaining a fairly good accuracy to reduce costs. AutoML experiments are more robust and their running times might be longer, in return, it includes useful machine learning features like model transparency and explainability.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
validation set for validation of hyperparameters
data is unbalanced --> look at better metrics, data featurization, data engineering
choose better model for hyperdrive
longer run time
grid search sampling, bayesian
use gpu compute and enable deep learning models, scikit learn estimator This estimator only supports single-node CPU training--> tensorflow estimator for parallel training (not precisely better result but faster provided the hardware resources are available)

A grid sampling is exhaustive, but more time-consuming. In contrast, a random sweep can get good results without taking as much time. (In most of the cases preferable when time efficiency is more important because the "improvement" through grid search is not significant.)
maybe Bayesian sampling better (selects values based on how previous values improved the training performance; some child runs showed the same accuracy at different parameter values so the whole sampling needn't be carried out completely.)Bayesian sampling does not support early termination. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
