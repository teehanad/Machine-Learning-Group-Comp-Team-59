import numpy
import pandas
import re
from sklearn.linear_model import LinearRegression
debug = 1

# This code is for the group competition in Kaggle for Computer Science Machine Learning at TCD.
# We are team59, the members are Lin, Deepthi, and Adam. Written by Lin Tung-Te 2019/11/6

# The files for this competition.
file_name_fit       = 'tcd-ml-1920-group-income-train.csv'
file_name_predict   = 'tcd-ml-1920-group-income-test.csv'
file_name_result    = 'tcd-ml-1920-group-income-submission.csv'


def preprocess(file_fit,file_predict):       #Adam's code
    #Import the file_fit
    #Before I begin working with new data I like to print informtation about the data to see what I am working with
    #Print some useful information about the data to see what we are dealing with
    print("----------------------------------------------------------------------------------------------------Adam's Notes----------------------------------------------------------------------------------------------------")
    print(""" 
    Set debug = 1 if you wish to see info about imported data printed, 0 if you do not want that \n
    I am not sure what crime rate is measured in terms of and what the upper and lower bounds of that scale are, seems fairly arbitrary to me \n
    Also housing Situation is redundant, its all zeros \n
    NaN in hair color makes sense as people can be bald so I shall leave that \n
    """)
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    if(debug == 1):   
        print("\n")
        print("\n")
        print("--------------------------------------------------------------------------------------------------TRAINING DATA INFO------------------------------------------------------------------------------------------------")
        file_fit.info()
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_fit.head(10))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(file_fit.describe())
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


    #Clean up begins
    #Remove the EUR at the end of Yearly Income in addition to Salary (e.g. Rental Income) using regex so data can be treated as a number
    #Change datatype of column to float64 instead of previous type of Object
    file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"] = file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"].replace(to_replace ='EUR', value = '', regex = True)
    file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"] = pandas.to_numeric(file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"])
    
    #Change #N/A to no so that there are not two options for no degree
    file_fit["University Degree"] = file_fit["University Degree"].replace( "#N/A" ,'no')

    file_predict["Yearly Income in addition to Salary (e.g. Rental Income)"] = file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"].replace(to_replace ='EUR', value = '', regex = True)
    file_predict["Yearly Income in addition to Salary (e.g. Rental Income)"] = pandas.to_numeric(file_fit["Yearly Income in addition to Salary (e.g. Rental Income)"])
    file_predict["University Degree"] = file_fit["University Degree"].replace( "#N/A" ,'no')

    return file_fit,file_predict 

def encoding(file_fit,file_predict):         #Deepthi's code
    return file_fit,file_predict

def analysis(file_fit,file_predict):         #Lin's code
    model = LinearRegression()
    fit_x = file_fit['Instance'].values.reshape(-1,1)
    fit_y = file_fit['Total Yearly Income [EUR]']
    model.fit(fit_x,fit_y)
    predict_x = file_predict['Instance'].values.reshape(-1,1)
    result = model.predict(predict_x)
    return result

def main():

    # Reading files.
    file_fit = pandas.read_csv(file_name_fit,low_memory=False)
    file_predict = pandas.read_csv(file_name_predict,low_memory=False)

    # Calling our functions.
    file_fit,file_predict = preprocess(file_fit,file_predict)
    file_fit,file_predict = encoding(file_fit,file_predict)
    result = analysis(file_fit,file_predict)

    # Output
    header = ['Instance','Total Yearly Income [EUR]']
    col1 = file_predict[header[0]]
    output = pandas.DataFrame({
            header[0]:col1,
            header[1]:result
    })
    
    output.to_csv(file_name_result,index=False)
    print('finished...')

if __name__ == "__main__":
    main()
