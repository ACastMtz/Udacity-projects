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
The AutoML classification task was run with similar parameters on the same dataset using **Accuracy** as the primary metric, as well as adding a couple more parameters such as **n_cross_validations** (number of cross validations to perform if validation data is not specified), **max_cores_per_iteration** (maximum number of threads per iteration, -1 uses all available cores), and **max_concurrent_iterations** (maximum number of iterations executed in parallel). As already mentioned, the **VotingEnsemble method** yielded the highest accuracy value. A voting ensemble is a machine learning model that builds predictions from combining results from other models, it is therefore sometimes refered to as a meta-model. More specifically, Azure AutoML uses the **PreFittedVotingClassifier Class** (inherited from sklearn *ensemble VotingClassifier* class) with the parameters following parameters:

  Estimators: Models to include in the voting classifier
  
  Weights: Weights for each estimator
  
  flatten_transform: Defines the way the results are displayed

## Pipeline comparison
\
As a first noticeable difference between the Hyperdrive and the AutoML runs is the higher accuracy provided by the latter method: **0.9073 Hyperdrive < 0.9178 AutoML**. One possible explanation for this is that with Hyperdrive only one model was trained, AutoMl on the other hand trained multiple models and then applied a voting classifier at the end. The advantage of the voting classifier is that it is very useful when the each of combined classifiers already show good performance and through combining them their indiviual weaknesses are reduced. 

Another difference is the time it took for each run to complete: **Hyperdrive appr. 10 min** and **AutoML appr. 20**. This is consistent due to AutoML's thourough search for a good model, it is expected that it takes longer to sweep multiple different models with different parameters, instead of just one custom-coded model as Hyperdrive does. However, one thing worth mentioning is that the fact that Hyperdrive uses a user-defined training script allows for a more direct control over the whole training process, making it easier to follow the pipeline and adjust if necessary to achieve the goal at hand; this could be making a highly accurate model or to expedite the training while maintaining a fairly good accuracy to reduce costs. AutoML experiments are more robust and their running times might be longer, in return, it includes useful machine learning features like model transparency and explainability.

## Future work
\
As with any ML project there is always ample room for improvements. I'll try to divide some of the possible steps to improve future experiments in a (hopefully) comprehensble way.

**Hardware**

Provided that enough financial resources are available and it would not represent a major issue if costs were to increment, the following hardware upgrades could be considered: To improve computational efficiency, faster computes could be used instead of the Standard_D2_V2. Compute clusters with more core nodes or even a GPU would boost the model's training speed (higher values in parameters max_cores_per_iteration and max_concurrent_iterations) and even make it feasible to use deep learning methods suchs as a **TensorFlow Estimator** (instead of a Sklearn estimator) or allowing TensorFlow models in the AutoML experiment. An increased efficiency would also be benefitial in cases where the run time is limited (e.g. using a VM from Udacity).

**Data**

AutoML provides extra information about the dataset that might have been overlooked (data guardrails). In our case the data seems to be unbalanced, this might lead to wrong assumptions about the model's metrics due to a high bias towards a class and a high-bias model is most likely to underfit the training data. The reason behind it is that our algorithms are supposed to maximize the accuracy (by reducing the error) and to accomplish this, it is best for most ML models to have balanced classes. Collecting more data (actual or synthetic data, resampling techniques) or doing some extra feature engineering might help fix this issue. 

**Metrics**

Even if the data weren't unbalanced, it is a good practice to look at multiple metrics to correctly evaluate a model. In the particular case of unbalanced data, **Accuracy** as the algorithm's primary metric is not the preferred performance measure to use for classifiers. Some common metrics that provide a better insight are: Confusion matrix, Precision, Recall, F1 Score. Choosing a "better" metric to maximize depends strongly on the context of the problem to solve. For example, if we are more interested in optimizing the detection of the *amount* of clients that might potentially subscribe to a term deposit with the bank, we might want to focus on the **Recall** and sacrifice the **Precision** (precision/recall tradeoff); on the other hand, if what is desired are clients that have a *very high probability* of subscribing then **Precision** would be a better metric to monitor.

**Algorithm**

Trying out multiple algorithms it's almost always a good approach to any machine learning problem. While this issue does not affect the AutoML run, the Hyperdrive uses a custom-coded model for the training. Using different models and not just different parameters could be beneficial (especially with umbalnced datasets), furthermore, increasing a model's complexity might reduce the bias. 

**Hyperparameters**

In the Hyperdrive experiment we used a random sampling method to find the "best" hyperparameters for the model. Alternatively, using a more exhaustive approach such as **Grid sampling** could lead to better hyperparameter values, albeit at a slower sweep speed. Another way could be using a **Bayesian sampler**, which selects values based on prior values that improved the performance. Some of the child runs from the Hyperdrive experiment showed the same accuracy at different parameter values so the whole sampling needn't be carried out completely, the downside is that Bayesian sampling does not support early termination.
Last but not least, a validation set could be added for validation of the hyperparameters instead of using cross validation on the training set.

## Resources
\
https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb

https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

https://machinelearningmastery.com/voting-ensembles-with-python/#:~:text=A%20voting%20ensemble%20(or%20a,model%20used%20in%20the%20ensemble

https://docs.microsoft.com/en-us/azure/machine-learning/?view=azure-ml-py

https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18

Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Edition, September 2019.
