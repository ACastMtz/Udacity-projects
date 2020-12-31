# Operationalizing Machine Learning

This project aims to provide an overview on how to create a cloud-based ML model and its production process from loading the dataset till consumption of the deployed model. Once again, the UCI Bank Marketing dataset was used to train a model to predict if a client is likely to subscribe to a term deposit with the bank.


## Architectural Diagram
![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/AZ_Arch_Diagram.png?raw=true)

## Key Steps
1. User Authentication: The first step is to provide the credentials and authentication required for logging into the Azure platform. However, this is not required (or even possible) when using the virtual lab offered by Udacity to work on the project. 

2. Dataset: Next step is to upload the Bank Marketing dataset to the Azure MK Studio so that it can be used to train the auto ML model.
![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/Dataset.png?raw=true)

3. Auto ML Model: Once the dataset is ready to be used, an automated ML run is created using *Classification* for the type of algorithm.

![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/automlrun_completed.png?raw=true)

After the run is completed, the best yielded model is a *Voting Ensemble*, which will be deployed in the following step.

![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/best_model.png?raw=true)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
