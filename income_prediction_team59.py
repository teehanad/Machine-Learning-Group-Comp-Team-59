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


    
    
    
#Chnaged the function to take in filetofix instead of file_fit and file_predict so that I didnt have to write each line of code twice, changed the main function to just call this function twice with each different file instead of what is was before
def preprocess(filetofix, filename): #Adam's code
    #Remove the EUR at the end of Yearly Income in addition to Salary (e.g. Rental Income) using regex so data can be treated as a number
    #Change datatype of column to float64 instead of previous type of Object
    filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"] = filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"].replace(to_replace ='EUR', value = '', regex = True)
    filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"] = pandas.to_numeric(filetofix["Yearly Income in addition to Salary (e.g. Rental Income)"])

    #All the below is replacing the first value passed into the function with the second one, should be able to see whats going on from the line itself
    filetofix["Year of Record"] = filetofix["Year of Record"].replace( numpy.NaN ,'None')

    filetofix["Housing Situation"] = filetofix["Housing Situation"].replace( "nA" ,'None')
    filetofix["Housing Situation"] = filetofix["Housing Situation"].replace( "0" ,'None')

    #Needs fixing
    filetofix["Work Experience in Current Job [years]"] = filetofix["Work Experience in Current Job [years]"].replace( "#NUM!" ,'0' )

    filetofix["Satisfation with employer"] = filetofix["Satisfation with employer"].replace( numpy.NaN ,'Unknown')

    filetofix["Gender"] = filetofix["Gender"].replace( "f" ,'female')
    filetofix["Gender"] = filetofix["Gender"].replace( "0" ,'None')
    filetofix["Gender"] = filetofix["Gender"].replace( numpy.NaN ,'None')

    filetofix["Country"] = filetofix["Country"].replace( '0' ,'Unknown')

    filetofix["University Degree"] = filetofix["University Degree"].replace( "#N/A" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( numpy.NaN ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "0" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "no" ,'None')
    filetofix["University Degree"] = filetofix["University Degree"].replace( "No" ,'None')

    filetofix["Hair Color"] = filetofix["Hair Color"].replace( '0' ,'Unknown')
    filetofix["Hair Color"] = filetofix["Hair Color"].replace( numpy.NaN ,'Unknown')

    #Debug Mode
    if debug == 1:
        print("\n")
        print("\n")
        print("------------------------------------------------------------------------------------------CHECK UNIQUE DATA FOR " +filename+"----------------------------------------------------------------------------------------")
        checkUnique(filetofix)
        print("----------------------------------------------------------------------------------------END CHECK UNIQUE DATA FOR " +filename+"---------------------------------------------------------------------------------------")



    return filetofix



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
    printInfo(file_fit, file_predict)
    file_fit = preprocess(file_fit, "FILE_FIT")
    file_predict = preprocess(file_predict, "FILE_PREDICT")
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
