# Import libraries
import numpy as np
import pandas as pd 
import argparse
from azureml.core.run import Run

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./Data/lending_club_loan.csv')


def data_process(df):
    # Mapping loan_satutus to binary values
    df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
    # Dropping NA or "insignificant" values
    df = df.drop('emp_title',axis=1)
    df = df.drop('emp_length',axis=1)
    df = df.drop('title',axis=1)
    fill_avg = df.groupby('total_acc').mean()['mort_acc']
    def fill_NA(x,y):
        """
        Fills in NA values of y with mean of x 
        """
        if np.isnan(y): 
            return fill_avg[x]
        else:
            return y
    df['mort_acc']=df.apply(lambda x: fill_NA(x['total_acc'],x['mort_acc']),axis=1)
    df = df.drop('grade',axis=1)
    df = df.drop('issue_d',axis=1)
    df = df.dropna()
    # Retrieving info from categorical data
    df['term'] = df['term'].apply(lambda x: int(x[:3]))
    df['zip_code'] = df['address'].apply(lambda x: x[-5:])
    df = df.drop('address',axis=1)
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
    df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
    drop_list = [
        'sub_grade',
        'verification_status',
        'application_type',
        'initial_list_status',
        'home_ownership',
        'purpose'
        ]
    dummies = pd.get_dummies(df[drop_list],drop_first=True)
    df = pd.concat([df.drop(drop_list,axis=1),dummies],axis=1)
    df = df.drop('loan_status',axis=1)
    x = df.drop('loan_repaid',axis=1)
    y = df['loan_repaid']

    return x,y

x,y = data_process(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

#   Normalizing
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    run.log('Regularization Strength:', np.float(args.C))
    run.log('Max iterations', np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    
    run.log('Accuracy', np.float(accuracy))

if __name__ == '__main__':
    main()