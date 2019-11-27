import numpy
import pandas
import re
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import train_test_split
#import lightgbm

from sklearn.ensemble import RandomForestRegressor

debug = 0

# This code is for the group competition in Kaggle for Computer Science Machine Learning at TCD.
# We are team59, the members are Lin, Deepthi, and Adam. Written by Lin Tung-Te 2019/11/6

# The files are for this competition.
file_name_fit       = 'tcd-ml-1920-group-income-train.csv'
file_name_predict   = 'tcd-ml-1920-group-income-test.csv'
file_name_result    = 'tcd-ml-1920-group-income-submission.csv'

# The files are processed data.
file_name_fit_processed =       'train-processed.csv'
file_name_predict_processed =   'test-processed.csv'


def add_noise(series, noise_level):
    #Fnction to add noise to the series
    return series * (1 + noise_level * numpy.random.randn(len(series)))

def target_encode(dataset,testset,target):
    #Function to preform Target encoding
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pandas.concat([dataset, target], axis=1)

    #Compute target mean 
    averages = temp.groupby(by=dataset.name)[target.name].agg(["mean", "count"])

    #Compute smoothing
    smoothing = 1 / (1 + numpy.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    #Apply average function to all target data
    prior = target.mean()

    #The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply average
    assert len(dataset) == len(target)
    assert dataset.name == testset.name
    temp = pandas.concat([dataset, target], axis=1)

    # Compute target mean 
    averages = temp.groupby(by=dataset.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + numpy.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to dataset and testset
    feature_dataset = pandas.merge(
        dataset.to_frame(dataset.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=dataset.name,
        how='left')['average'].rename(dataset.name + '_mean').fillna(prior)

    # pandas.merge does not keep the index so restore it
    feature_dataset.index = dataset.index 
    feature_testset = pandas.merge(
        testset.to_frame(testset.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=testset.name,
        how='left')['average'].rename(dataset.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    feature_testset.index = testset.index

    return add_noise(feature_dataset, noise_level), add_noise(feature_testset, noise_level)

def checkUnique(filetofix): #Helper function made by Adam
    print("Year of Record", filetofix["Year of Record"].unique())
    print("Housing Situation", filetofix["Housing Situation"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------") 
    print("Crime Level in the City of Employement", filetofix["Crime Level in the City of Employement"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Work Experience in Current Job [years]", filetofix["Work Experience in Current Job [years]"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Satisfation with employer", filetofix["Satisfation with employer"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Gender", filetofix["Gender"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Age", filetofix["Age"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Country", filetofix["Country"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Size of City", filetofix["Size of City"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Profession", filetofix["Profession"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("University Degree", filetofix["University Degree"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Wears Glasses", filetofix["Wears Glasses"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Hair Color", filetofix["Hair Color"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Body Height",filetofix["Body Height [cm]"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Additional income", filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"].unique())
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Total income", filetofix["Total Yearly Income [EUR]"].unique())
    return


def printInfo(file_fit,file_predict): #Helper function made by Adam
    #Before I begin working with new data I like to print informtation about the data to see what I am working with
    #Print some useful information about the data to see what we are dealing with
    print("---------------------------------------------------------------------------------------------------ADAM'S NOTES----------------------------------------------------------------------------------------------------")
    print(""" 
    Set debug = 1 if you wish to see info about imported data printed, 0 if you do not want that \n
    I am not sure what crime rate is measured in terms of and what the upper and lower bounds of that scale are, seems fairly arbitrary to me \n
    Unknown in hair color makes sense as people can be bald so I shall leave that \n
    """)
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    if debug == 1:   
        print("\n")
        print("\n")
        print("--------------------------------------------------------------------------------------------------TRAINING DATA INFO------------------------------------------------------------------------------------------------")
        file_fit.info()
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_fit.head(10))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_fit.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Shape:", file_fit.shape)
        print("------------------------------------------------------------------------------------------------END TRAINING DATA INFO ---------------------------------------------------------------------------------------------")

        print("\n")
        print("\n")
        print("---------------------------------------------------------------------------------------------------TEST DATA INFO --------------------------------------------------------------------------------------------------")
        file_predict.info()
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_predict.head(10))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_predict.describe())
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Shape:" ,file_predict.shape)
        print("--------------------------------------------------------------------------------------------------END TEST DATA INFO------------------------------------------------------------------------------------------------")

    
#Chnaged the function to take in filetofix instead of file_fit and file_predict so that I didnt have to write each line of code twice, changed the main function to just call this function twice with each different file instead of what is was before
def preprocess(filetofix, filename): #Adam's code
   
    #Remove the EUR at the end of Yearly Income in addition to Salary (e.g. Rental Income) using regex so data can be treated as a number
    #Change datatype of column to float64 instead of previous type of Object
    filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"] = filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"].replace(to_replace ='EUR', value = '', regex = True)
    filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"] = pandas.to_numeric(filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"])

    filetofix['Year of Record'].fillna(filetofix['Year of Record'].median(), inplace=True)  # Use median to replace of 'NA'. By Lin 2019/11/10

    filetofix["Housing Situation"] = filetofix["Housing Situation"].replace( "nA" ,'None')
    filetofix["Housing Situation"] = filetofix["Housing Situation"].replace( "0" ,'None')

    filetofix["Work Experience in Current Job [years]"] = filetofix["Work Experience in Current Job [years]"].replace( "#NUM!" ,numpy.NaN)
    #print('median:',filetofix['Work Experience in Current Job [years]'].median())
    #filetofix['Work Experience in Current Job [years]'].fillna(filetofix['Work Experience in Current Job [years]'].median(), inplace=True)
    filetofix['Work Experience in Current Job [years]'].fillna(0, inplace=True)
    filetofix["Work Experience in Current Job [years]"] = pandas.to_numeric(filetofix["Work Experience in Current Job [years]"])

    filetofix["Gender"] = filetofix["Gender"].replace( "f" ,'female')
    filetofix["Gender"] = filetofix["Gender"].replace( "0" ,'other')
    filetofix["Gender"] = filetofix["Gender"].replace( "nan" ,numpy.NaN)
    filetofix["Gender"] = filetofix["Gender"].replace( 'unknown' ,numpy.NaN)
    filetofix["Gender"] = filetofix["Gender"].replace( "#N/A" ,numpy.NaN)
    filetofix['Gender'].fillna(filetofix['Gender'].mode(dropna=True)[0], inplace=True) 

    filetofix["Satisfation with employer"] = filetofix["Satisfation with employer"].replace( "nan" ,numpy.NaN)
    filetofix['Satisfation with employer'].fillna(filetofix['Satisfation with employer'].mode(dropna=True)[0], inplace=True) 


    filetofix["Country"] = filetofix["Country"].replace( '0' ,'Unknown')

    filetofix["University Degree"] = filetofix["University Degree"].replace( "#N/A" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( numpy.NaN ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "0" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "no" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "No" ,'None')

    filetofix["Hair Color"] = filetofix["Hair Color"].replace( '0' ,'Unknown')
    filetofix["Hair Color"] = filetofix["Hair Color"].replace( numpy.NaN ,'Unknown')

    filetofix['Profession'].fillna(filetofix['Profession'].mode(dropna=True)[0], inplace=True)
    filetofix['Country'].fillna(filetofix['Country'].mode(dropna=True)[0], inplace=True) 

    #removed_columns = ['Crime Level in the City of Employement', 'Size of City','Wears Glasses','Yearly Income in addition to Salary (e.g. Rental Income)']
    #removed_columns = ['Crime Level in the City of Employement', 'Hair Color']
    removed_columns = ['Age','Hair Color','Wears Glasses','Body Height [cm]']
    for column in removed_columns:
        del filetofix[column]

    return filetofix

def encoding(file_fit,file_predict):         #Deepthi's code
    #One hot encoding is here, because Deepthi doesn't know how to put her code in this function, therefore I modified her code and put here. By Lin 2019/11/9
    print('Start one hot encoding.')
    # The columns used to one hot encoding.
    #columns = ['Gender','Housing Situation','Hair Color','University Degree','Satisfation with employer']
    columns = []
    for column in columns:
        if column in file_fit.columns:
            file_fit = pandas.get_dummies(file_fit,columns=[column],prefix=[column])
            file_predict = pandas.get_dummies(file_predict,columns=[column],prefix=[column])
    print('One hot encoding is finished.')

    # The columns used to target encoding.
    columns = ['Gender','Housing Situation','University Degree','Satisfation with employer','Country','Profession','Hair Color']
    target = file_fit['Total Yearly Income [EUR]']
    print('Start target encoding.')
    for column in columns:
        if column in file_fit.columns:
            file_fit[column],file_predict[column] = target_encode(file_fit[column],file_predict[column],target)
    print('Target encoding is finished.')
    return file_fit,file_predict

def analysis(file_fit,file_predict,model_no,test_mode):         #Lin's code updated:2019/11/10

##    print('Output the processed files.')
##    file_fit.to_csv(file_name_fit_processed,index=False)
##    file_predict.to_csv(file_name_predict_processed,index=False)

##    print('Check income.')
##    null_columns=file_fit.columns[file_fit.isnull().any()]
##    print(file_fit[file_fit["Total Yearly Income [EUR]"].isnull()][null_columns])

    fit_y = file_fit['Total Yearly Income [EUR]']

    #The columns are not used temporarily.
    #unused_columns = ['Total Yearly Income [EUR]','Satisfation with employer','Country','Profession']
    unused_columns = ['Total Yearly Income [EUR]']
    for column in unused_columns:
        file_fit = file_fit.drop([column],axis=1)
        file_predict = file_predict.drop([column],axis=1)

    
##    improved_columns = ['Country','Yearly Income in addition to Salary (e.g. Rental Income)']
##    for column in improved_columns:
##        file_fit[column] = pow(file_fit[column],0.5)
    
    # Do not include instance.
    fit_x = file_fit.iloc[:,1:]
    predict_x = file_predict.iloc[:,1:]


    if(test_mode):
        fit_x,predict_x,fit_y,predict_y = train_test_split(fit_x.values, fit_y.values, test_size=0.2, random_state=30)
    else:
        fit_x = fit_x.values
        fit_y = fit_y.values


    if (model_no == 0):
        print('Using linear regression')
        model = make_pipeline(PolynomialFeatures(1),linear_model.LinearRegression()) #using PolynomialFeatures
        print('Start fitting.')
        model.fit(fit_x,fit_y)
        print('Fitting is finished, start predicting.')
        result = model.predict(predict_x)
        print('Predicting is finished.')
    elif(model_no == 1):
        print('Using LightGBM')
        train_x, test_x, train_y, test_y = train_test_split(fit_x, fit_y, test_size=0.2, random_state=42)
        
        train_data = lightgbm.Dataset(train_x, label=train_y)
        test_data = lightgbm.Dataset(test_x, label=test_y)

        parameters = {
            'objective': 'regression',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.1,
            'verbose': 0
        }

        model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=8000
                           #,early_stopping_rounds=600
                               ,early_stopping_rounds=100
                               )
        
        model.save_model('gbm_model.txt')
        model = lightgbm.Booster(model_file='gbm_model.txt')
        
        result = model.predict(predict_x)

    elif(model_no == 2):
        print("Using RandomForest")
        randomForestGenerator = RandomForestRegressor(n_estimators=200,n_jobs=-1)
       
        #Fit the model
        randomForestGenerator.fit(fit_x,fit_y)
       
        #Making predictions
        print("Starting prediction")
        result = randomForestGenerator.predict(predict_x)

    elif(model_no == 3):
        print('Using BayesianRidge')
        model = BayesianRidge()
        model.fit(fit_x,fit_y)
        result = model.predict(predict_x)
        

        
    if(test_mode):
        score = result-predict_y
        score = abs(score)
        print('score:',sum(score))
    else:   
        # Output
        print('Output the files.')
        header = ['Instance','Total Yearly Income [EUR]']
        col1 = file_predict[header[0]]
        output = pandas.DataFrame({
                header[0]:col1,
                header[1]:result
        })
        output.to_csv(file_name_result,index=False)

def main():

    # Reading files.
    file_fit = pandas.read_csv(file_name_fit,low_memory=False)
    file_predict = pandas.read_csv(file_name_predict,low_memory=False)

    
    print(file_fit.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
    print(file_predict.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))

    # Calling our functions.
    printInfo(file_fit, file_predict)
    
    file_fit = preprocess(file_fit, "FILE_FIT")
    file_predict = preprocess(file_predict, "FILE_PREDICT")
    
    file_fit,file_predict = encoding(file_fit,file_predict)

    analysis(file_fit,file_predict,2,False)

    print('Finished...')

if __name__ == "__main__":
    main()
