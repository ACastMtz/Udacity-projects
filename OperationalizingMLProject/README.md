# Operationalizing Machine Learning

This project aims to provide an overview on how to create a cloud-based ML model and its production process: from loading the dataset till consumption of the deployed model. Once again, the UCI Bank Marketing dataset was used to train a model to predict if a client is likely to subscribe to a term deposit with the bank.


## Architectural Diagram
![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/AZ_Arch_Diagram.png?raw=true)

## Key Steps
1. User Authentication: The first step is to provide the credentials and authentication required for logging into the Azure platform. However, this is not required (or even possible) when using the virtual lab offered by Udacity to work on the project. 

2. Dataset: Next step is to upload the Bank Marketing dataset to the Azure ML Studio so that it can be used to train the auto ML model.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/Dataset.png?raw=true)

3. Auto ML Model: Once the dataset is ready to be used, an automated ML run is created using *Classification* for the type of task to train several models using different combinations of algorithms and hyperparameters. *Accuracy* was chosen as the primary metric for the run.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/automlrun_completed.png?raw=true)

  After the run is completed, the best yielded model is a *VotingEnsemble*, which will be deployed in the following step.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/best_model.png?raw=true)

4. Deployment: In order to be consumed, the model needs to be first deployed. This can be achieved with an Azure Container Instance (ACI), thus exposing the model.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/deploy_aidisabled.png?raw=true)

5. Logging: After deployment, a REST endpoint is created which allows interaction with the HTTP API service over POST requests. A helpful feature within the ACI is *Application Insights* for debugging and troubleshooting in production environments by making it possible to retrieve logs from the deployed model.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/deploy_aienabled.png?raw=true)
  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/logging.png?raw=true)

6. Consumption: To consume the model a Docker container serving *Swagger* and running locally is downloaded. The swagger instance contains the docuemntation for the HTTP API of the model.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/swagger.png?raw=true)
  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/swagger_1.png?raw=true)

* The interaction is carried out by feeding JSON strings inside the python script *endpoints.py* to the Swagger server. By sending HTTP POST requests to the server's endpoint and the JSON payload, the model can make predictions and send them back as HTTP POST responses.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/endpoints_int.png?raw=true)

* When deploying a model it is important to get an idea of an acceptable performance for the HTTP API to keep track of the model's functioning. This is also known as benchmarking and it is an importatnt step of a MLOps environment. In this project a benchmark is created for the  Azure container hosting the deployed model using *Apache Benchmark* which leverages the REST APIs reponses to monitor the model's performance.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/benchmark.png?raw=true)

7. Pipeline: Finally, a pipeline with an auto ML step run is created, published and consumed using the python SDK and a jupyter notebook. Pipelines are helpful for automation of different tasks in a workflow. Using the python SDK to automate creation, publishing and consumption of a model is a great tool to make every part of the process be more productive, resilient and scalable.

  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/Pipeline_running.png?raw=true)
  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/ds_automlmod.png?raw=true)
  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/widget.png?raw=true)
  ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/published_pipeline.png?raw=true)



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
As stated in the project's <a href="https://review.udacity.com/#!/rubrics/2893/view">Rubrics</a> a written description of the recording will be provided instead of audio.

## Outlook
Even though the porpose of this project is not to achieve an excellent predictive model but work through the basics of MLOps, there is still, as it is usually the case, room for future improvements:

-> Extending training time

-> Training with batch and online learning

-> Allowing Deep Learning models to be considered in the Auto ML runs

-> Boosting the training speed (assuming also that DL models are being used) by enabling dedicated hardware such as GPUs or TPUs

->...
