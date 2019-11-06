import numpy
import pandas
from sklearn.linear_model import LinearRegression

# This code is for the group competition in Kaggle for Computer Science Machine Learning at TCD.
# We are team59, the members are Lin, Deepthi, and Adam. Written by Lin Tung-Te 2019/11/6

# The files for this competition.
file_name_fit       = 'tcd-ml-1920-group-income-train.csv'
file_name_predict   = 'tcd-ml-1920-group-income-test.csv'
file_name_result    = 'tcd-ml-1920-group-income-submission.csv'


def preprocess(file_fit,file_predict):       #Adam's code
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
