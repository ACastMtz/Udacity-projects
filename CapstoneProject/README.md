# Loan Default Prediction

This is the **Capstone Project** from Udacity's Nanodegree *Machine Learning Engineer with Microsoft Azure Nanodegree* and its objective is to apply the acquired knowledge to build a ML model and deploy it using an external dataset (i.e. not included in the Azure environment) and in that way emulate a somewhat more realistic scenario.

We will be using the **LendingClub** dataset for borrower failure risk analysis, that is, analysing the factors that contribute to increase or decrease the danger that a *loaner* may default on repaying a loan to the *lender*. The overall goal is to learn how real world problems can be solved, particularly in the context of risk analytics in banking and financial services, using Machine Learning.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

### Prerequisites
To be able to follow along this project on remote compute resources the following steps are needed:

1. Log into your Azure account
2. Create a Workspace
3. Acces the ML Studio
4. Clone or download the notebooks and scripts from this repo: 'hyperparameter_tuning.ipynb', 'automl.ipynb', 'train.py' and 'endpoint.py', and upload them to your working folder in the Studio
5. Download the **LendingClub** dataset from Kaggle [here](https://www.kaggle.com/wordsforthewise/lending-club) and upload it to the Studio
6. [OPTIONAL] Create a folder called "Data" and move the dataset there. The notebooks access the data from this folder, if there is no "Data" folder the code in the notebooks and the 'train.py' script will have to be changed to the path where the dataset is
7. Create a compute instance to run the notebooks
8. Open the jupyter notebooks and run the cells 

### Project Overview
We will train and optimize an Azure ML pipeline using two different methods: **Hyperparameter Tuning using Azure's Hyperdrive** and **Auto Machine Learning**. For the Hyperdrive run, a simple ligistic regression model was chosen. The best model from both approaches is then deployed to be interacted with.

Below is an overview of the workflow:


.center[ ![My image](./Images/capstone-diagram.png) .caption[**Fig. 1:** Image caption]]


## Dataset

### Overview

In this project, the *lending club* dataset from the LendingClub American peer-to-peer lending company was used. The purpose is to use data for risk analytics and minimization in a banking and financial context. To achieve that, statistical information about past loan applicants is used to build a model using supervised learning, where the labels are whether or not the applicant failed to fully repay the loan, to be able to predict if a new applicant is likely to repay the loan. The aim is for the model to identify patterns in the dataset that can be used to determine the outcome of the new application based on the financial history of the applicant. In this way, the probability of defaulting the loan can be assessed and lenders can make an informed decision accordingly that may reduce the loss of business for the company by cutting down the credit loss, e.g. by denying the loan, raising interest rates, offering a different loan amount, etc.

Below is a table with all the information available in the dataset for training the model.
        
   |      LoanStatNew     | Description                                                                                               |
   |:--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
   | loan_amnt            | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. |
   | term                 | The number of payments on the loan. Values are in months and can be either 36 or 60.  |
   | int_rate             | Interest Rate on the loan                                                                                                                                                                                |
   | installment          | The monthly payment owed by the borrower if the loan originates.                                                                                                                                         |
   | grade                | LC assigned loan grade                                                                                                                                                                                   |
   | sub_grade            | LC assigned loan subgrade                                                                                                                                                                                |
   | emp_title            | The job title supplied by the Borrower when applying for the loan.*                                                                                                                                      |
   | emp_length           | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.                                                                        |
   | home_ownership       | The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER                                                    |
   | annual_inc           | The self-reported annual income provided by the borrower during registration.                                                                                                                            |
   | verification_status  | Indicates if income was verified by LC, not verified, or if the income source was verified                                                                                                               |
   | issue_d              | The month which the loan was funded                                                                                                                                                                      |
   | loan_status          | Current status of the loan                                                                                                                                                                               |
   | purpose              | A category provided by the borrower for the loan request.                                                                                                                                                |
   | title                | The loan title provided by the borrower                                                                                                                                                                  |
   | zip_code             | The first 3 numbers of the zip code provided by the borrower in the loan application.                                                                                                                    |
   | addr_state           | The state provided by the borrower in the loan application                                                                                                                                               |
   | dti                  | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income. |
   | earliest_cr_line     | The month the borrower's earliest reported credit line was opened                                                                                                                                        |
   | open_acc             | The number of open credit lines in the borrower's credit file.                                                                                                                                           |
   | pub_rec              | Number of derogatory public records                                                                                                                                                                      |
   | revol_bal            | Total credit revolving balance                                                                                                                                                                           |
   | revol_util           | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.                                                                               |
   | total_acc            | The total number of credit lines currently in the borrower's credit file                                                                                                                                 |
   | initial_list_status  | The initial listing status of the loan. Possible values are – W, F                                                                                                                                       |
   | application_type     | Indicates whether the loan is an individual application or a joint application with two co-borrowers                                                                                                     |
   | mort_acc             | Number of mortgage accounts.                                                                                                                                                                             |
   | pub_rec_bankruptcies | Number of public record bankruptcies   |

### Task
The task at hand is, not only to train an accurate predicitive model using a logistic regression algorithm, but also to gain an insight into the most important features that determine the result yielded by the model. This allows the company to understand which variables are strong indicators of loan default and apply this knowledge in future risk assessment.

### Access
The dataset can be found and downloaded from Kaggle [here](https://www.kaggle.com/wordsforthewise/lending-club). It was then uploaded to the work folder and accessed locally.
## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
