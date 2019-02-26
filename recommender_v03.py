from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import scipy.stats as ss
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
# test results in a csv file should be all of the same ML algorithm
directory = "./testdat/used/"
data = {}
models = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        test_data = pd.read_csv(directory+filename, sep = ",")
        alg = test_data.iloc[0,len(test_data.columns) - 5:len(test_data.columns) - 3]
        alg = alg[0]+":"+alg[1] # get algorithm name and library
        if (alg not in data):
            models[alg] = {}
            data[alg] = {}
            # collecting data
            predictors = test_data.iloc[:,len(test_data.columns) - 2:len(test_data.columns)] # num_pred and num_obs
            responses = test_data.iloc[:,0:len(test_data.columns) - 6].\
                                            join(test_data.iloc[:,len(test_data.columns) - 3]) # all the parameters we're making a model of
            data[alg] = (predictors,responses)
        else:
            data[alg][0].append( test_data.iloc[:,len(test_data.columns) - 2:len(test_data.columns)]) # num_pred and num_obs
            data[alg][1].append( test_data.iloc[:,0:len(test_data.columns) - 6].\
                            join(test_data.iloc[:,len(test_data.columns) - 3])) # all the parameters we're making a model of

       # print(predictors.columns.values)

       # print(responses.columns.values)

    else:
        continue

# make models
for alg, dat in data.items():
    for response in dat[1]: # collecting data
        #column_name = responses[response]
        predictorTrain, predictorTest, responseTrain, responseTest = \
        train_test_split(dat[0], dat[1][response], test_size=0.2, random_state=1)
        lm = LinearRegression()
        try:
            modlm = lm.fit(predictorTrain, responseTrain)
        except ValueError:
            print (response)
            print (predictorTrain)
            print (responseTrain)
        print (alg + " ("+response+") "+" accuracy :" + str(modlm.score(predictorTest, responseTest)))
        models[alg][response] = modlm


accuracy_weight = 10
response_time_weight = 2
CPU_utilization_weight = 4
IO_operations_weight = 6
Physical_reads_weight = 3

n_observations = 1000000
n_predictors = 1100

def recommend():
    for alg, responses in models.items():
        for response, model in responses.items():
            pred = model.predict([[n_observations,n_predictors]])
            #print (response+":"+str(pred))
            if pred == 0:
                res = float("infinity")
            else:
                res = 1/pred
            print (response+":"+str(res))
            #print(res)
recommend()
