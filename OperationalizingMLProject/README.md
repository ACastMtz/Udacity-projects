# Operationalizing Machine Learning

This project aims to provide an overview on how to create a cloud-based ML model and its production process: from loading the dataset till consumption of the deployed model. Once again, the UCI Bank Marketing dataset was used to train a model to predict if a client is likely to subscribe to a term deposit with the bank.


## Architectural Diagram
![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/AZ_Arch_Diagram.png?raw=true)

## Key Steps
1. User Authentication: The first step is to provide the credentials and authentication required for logging into the Azure platform. However, this is not required (or even possible) when using the virtual lab offered by Udacity to work on the project. 

2. Dataset: Next step is to upload the Bank Marketing dataset to the Azure ML Studio so that it can be used to train the auto ML model.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/Dataset.png?raw=true)

3. Auto ML Model: Once the dataset is ready to be used, an automated ML run is created using *Classification* for the type of task to train several models using different combinations of algorithms and hyperparameters. *Accuracy* was chosen as the primary metric for the run.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/automlrun_completed.png?raw=true)

* After the run is completed, the best yielded model is a *VotingEnsemble*, which will be deployed in the following step.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/best_model.png?raw=true)

4. Deployment: In order to be consumed, the model needs to be first deployed. This can be achieved with an Azure Container Instance (ACI), thus exposing the model.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/deploy_aidisabled.png?raw=true)

5. Logging: After deployment, a REST endpoint is created which allows interaction with the HTTP API service over POST requests. A helpful feature within the ACI is *Application Insights* for debugging and troubleshooting in production environments by making it possible to retrieve logs from the deployed model.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/deploy_aienabled.png?raw=true)
 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/logging.png?raw=true)

6. Consumption: To consume the model a Docker container serving *Swagger* and running locally is downloaded. The swagger instance contains the docuemntation for the HTTP API of the model.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/swagger.png?raw=true)
 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/swagger_1.png?raw=true)

* The interaction is carried out by feeding JSON strings inside the python script *endpoints.py* to the Swagger server. By sending HTTP POST requests to the server's endpoint and the JSON payload, the model can make predictions and send them back as HTTP POST responses.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/endpoints_int.png?raw=true)

* When deploying a model it is important to get an idea of an acceptable performance for the HTTP API to keep track of the model's functioning. This is also known as benchmarking and it is an importatnt step of a MLOps environment. In this project a benchmark is created for the  Azure container hosting the deployed model using *Apache Benchmark* which leverages the REST APIs reponses to monitor the model's performance.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/benchmark.png?raw=true)

7. Pipeline: Finally, a pipeline with an auto ML step run is created, published and consumed using the python SDK and a jupyter notebook (see aml-pipelines-with-automated-machine-learning-step.ipynb in the project's repository). Pipelines are helpful for automation of different tasks in a workflow. Using the python SDK to automate creation, publishing and consumption of a model is a great tool to make every part of the process be more productive, resilient and scalable.

 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/Pipeline_running.png?raw=true)
 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/ds_automlmod.png?raw=true)
 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/widget.png?raw=true)
 * ![alt text](https://github.com/ACastMtz/Udacity-projects/blob/main/OperationalizingMLProject/Images/published_pipeline.png?raw=true)

## Screen Recording
As stated in the project's <a href="https://review.udacity.com/#!/rubrics/2893/view">Rubrics</a> a written description of the recording will be provided instead of audio.

The video can be found following <a href="https://youtu.be/Rn81BkKon_A">this link</a>.


### General Description

* To make it feasible to describe the main topics of the project within a 5 min long video, the startpoint is after the dataset was uploaded and the auto ML model was trained and deployed. Firstly, it shows a deployed working ML model and the initial steps needed to interact with the model hosted by the ACI. Then it shows the model available on the swagger-ui. Afterwards, different HTTP requests are sent to interact with the sever. Then benchmarking follows and finally the pipeline published using a jupyter notebook. Please note that by the end of the video, the second submission step for the pipeline is still running due to time limitations from the virtual lab provided by Udacity.

* Script:

 * * [0:00 - 1:10] After uploading the UCI Bank Marketing dataset, I ran an autoML experiment which yielded a VotingEnsemble as the best trained model. This model was then deployed on an Azure Container Instance and here we can see Run 1 being successfully completed and under "Deploy status" the name of the best model deployment. By clicking on this link we can find the information of the deployed model so that we can copy the REST enpoint URL needed in the next steps. For consumption we also need the primary key, which can be found in the Consume section. We then updated the URL and the key in the endpoint file which is the python script that is required to interact with the HTTP API. The same process has to be done for the benchmark file which is the shell file for benchmarking.
 
 * * [1:11 - 2:10] We then run the swagger.sh file to download a swagger instance, which is running locally, to allow us to consume the model. Please note that the swagger.json file from the deployed modeled has already been downloaded and placed in the same folder with all the other files needed for swagger to function properly. After this is done, we access the instance locally. Note that it is running on port 9000. We then run the serve.py python script to serve our model to the swagger instance by creating a HTTP server to expose the downloaded swagger.json file for the local swagger-ui to use it. We then go to to local port 9000 where we can see the deployed autoML model available on the swagger-ui.
 
 * * [2:11 - 2:49] Then we run the endpoint.py script to feed API requests to the server's endpoint with a JSON payload. We then get a reponse that can be seen here on the command line console and if we change the information of one of the JSON strings from the endpoint.py scripy we could get a different response, which we do as we can see here.
 
 * * [2:50 - 3:04] The last part here is to benchmark the API. To do so, we run the benchmark.sh script. The script runs Apache Benchmark 10 times and here we get the information of the requests. We can see here the time it took for the API to process the 10 requests but also the average time per request along with a lot more details.
 
 * * [3:05 - 4:31] We go back to the Azure Machine Learning studio where we will use a jupyter notebook to create, deploy and consume a pipeline. We run through these initiall cells which are in charge of the first steps like uploading the dataset and determining the parameters for the autoML run. We then create a pipeline with an autoMLStep, we submit the pipeline experiment and wait till is done running it. We now see that the run is completed, run52, we go back to the notebook to examine the results. We retrieve the metrics and the best model, this might take a couple of seconds, and we also show the best model steps. Then we publish the pipeline to enable a REST Endpoint. We then get the REST url from the endpoint of the published pipeline and make a request to trigger a second run, which we can see running in the pipeline section of the Azure Machine Learning Studio.


## Outlook
Even though the porpose of this project is not to achieve an excellent predictive model but work through the basics of MLOps, there is still, as it is usually the case, room for future improvements:

-> Extending training time

-> Training with batch and online learning

-> Allowing Deep Learning models to be considered in the Auto ML runs

-> Boosting the training speed (assuming also that DL models are being used) by enabling dedicated hardware such as GPUs or TPUs

->...
