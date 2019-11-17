#!/usr/bin/env python
# coding: utf-8

# In[39]:


from sklearn import preprocessing
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
dataset = pd.read_csv("tcd-ml-1920-group-income-train.csv",na_values=['#NUM!'])
testset = pd.read_csv("tcd-ml-1920-group-income-test.csv",na_values=['#NUM!'])


# In[40]:


def preprocessing(dataset,testset):
    #Dropping features
    
    dataset=dataset.drop(['Work Experience in Current Job [years]'],axis=1)
    dataset=dataset.drop(['Yearly Income in addition to Salary (e.g. Rental Income)'],axis=1)
    
   
    testset=testset.drop(['Work Experience in Current Job [years]'],axis=1)
    testset=testset.drop(['Yearly Income in addition to Salary (e.g. Rental Income)'],axis=1)
         
   
    #Replacing missing values
    dataset['Year of Record'].fillna(dataset['Year of Record'].median(), inplace=True)
    dataset['Year of Record']=dataset['Year of Record'].replace(['#N/A'],'0')
    dataset['Housing Situation'].fillna(dataset['Housing Situation'], inplace=True)
    dataset['Housing Situation'] = dataset['Housing Situation'].str.replace(" ","")
    dataset['Crime Level in the City of Employement'].replace('', np.nan, inplace=True)
    dataset['Crime Level in the City of Employement'].fillna(dataset['Crime Level in the City of Employement'].median(),inplace=True)
    dataset['Satisfation with employer'].fillna(dataset['Satisfation with employer'], inplace=True)   
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].str.replace(" ","")
    dataset['Gender'] = dataset['Gender'].replace(['0','nan' ], 'unknown') 
    dataset['Gender'] = dataset['Gender'].replace(['f'], 'female')
    dataset['University Degree'] = dataset['University Degree'].replace(['0','nan'], 'unknown')
    dataset['University Degree']=dataset['University Degree'].replace(['#N/A'],'0')
    dataset['Hair Color'] = dataset['Hair Color'].replace(['0','nan'], 'Unknown')
    dataset['Profession'] = dataset['Profession'].str.replace(" ","") 
    dataset['Country'] = dataset['Country'].str.replace(" ","")
    
    testset['Year of Record'].fillna(testset['Year of Record'].median(), inplace=True)
    testset['Year of Record']=testset['Year of Record'].replace(['#N/A'],'0')
    #testset['Housing Situation'].fillna(testset['Housing Situation'], inplace=True)
    #testset['Housing Situation'] = testset['Housing Situation'].str.replace(" ","")
    testset['Crime Level in the City of Employement'].replace('', np.nan, inplace=True)
    testset['Crime Level in the City of Employement'].fillna(testset['Crime Level in the City of Employement'].median(),inplace=True)
    testset['Satisfation with employer'].fillna(dataset['Satisfation with employer'], inplace=True)   
    testset['Satisfation with employer'] = dataset['Satisfation with employer'].str.replace(" ","")
    testset['Gender'] = testset['Gender'].replace(['0','nan' ], 'unknown') 
    testset['Gender'] = testset['Gender'].replace(['f'], 'female')
    testset['University Degree'] = testset['University Degree'].replace(['0','nan'], 'unknown')
    testset['University Degree']=testset['University Degree'].replace(['#N/A'],'0')
    testset['Hair Color'] = testset['Hair Color'].replace(['0','nan'], 'Unknown')
    testset['Profession'] = testset['Profession'].str.replace(" ","") 
    testset['Country'] = testset['Country'].str.replace(" ","")
    return dataset,testset
    


# In[35]:


def one_hot_encoding(dataset,testset):
#DATASET    
    #dataset=pd.get_dummies(dataset,columns=['Gender'],prefix=['Gender'])
    dataset=pd.get_dummies(dataset,columns=['Housing Situation'],prefix=['Housing Situation'])
    #dataset=pd.get_dummies(dataset,columns=['Hair Color'],prefix=['Hair Color']) 
    dataset=pd.get_dummies(dataset,columns=['Satisfation with employer'],prefix=['Hair Color']) 
    dataset=pd.get_dummies(dataset,columns=['Year of Record'],prefix=['Year of Record']) 
    #dataset=pd.get_dummies(dataset,columns=['University Degree'],prefix=['University Degree'])
    #dataset=pd.get_dummies(dataset,columns=['Profession'],prefix=['Profession'])
    #dataset=pd.get_dummies(dataset,columns=['Country'],prefix=['Country'])


#TESTSET
    
    #testset=pd.get_dummies(testset,columns=['Gender'],prefix=['Gender'])
    testset=pd.get_dummies(testset,columns=['Housing Situation'],prefix=['Housing Situation'])
    #testset=pd.get_dummies(testset,columns=['Hair Color'],prefix=['Hair Color']) 
    testset=pd.get_dummies(testset,columns=['Satisfation with employer'],prefix=['Hair Color']) 
    testset=pd.get_dummies(testset,columns=['Year of Record'],prefix=['Year of Record']) 
    #testset=pd.get_dummies(testset,columns=['University Degree'],prefix=['University Degree']) 
    #testset=pd.get_dummies(testset,columns=['Profession'],prefix=['Profession'])
    #testset=pd.get_dummies(dataset,columns=['Country'],prefix=['Country'])
    return dataset,testset    


# In[41]:


def add_noise(series, noise_level):
    #Fnction to add noise to the series
    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(dataset,testset,target):
    #Function to preform Target encoding
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pd.concat([dataset, target], axis=1)

    #Compute target mean 
    averages = temp.groupby(by=dataset.name)[target.name].agg(["mean", "count"])

    #Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    #Apply average function to all target data
    prior = target.mean()

    #The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply average
    assert len(dataset) == len(target)
    assert dataset.name == testset.name
    temp = pd.concat([dataset, target], axis=1)

    # Compute target mean 
    averages = temp.groupby(by=dataset.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to dataset and testset
    feature_dataset = pd.merge(
        dataset.to_frame(dataset.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=dataset.name,
        how='left')['average'].rename(dataset.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    feature_dataset.index = dataset.index 
    feature_testset = pd.merge(
        testset.to_frame(testset.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=testset.name,
        how='left')['average'].rename(dataset.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    feature_testset.index = testset.index

    return add_noise(feature_dataset, noise_level), add_noise(feature_testset, noise_level)


# In[42]:


def model_predict(dataset,testset):
    income=dataset['Total Yearly Income [EUR]']
    dataset['Gender'],testset['Gender'] = target_encode(dataset['Gender'],testset['Gender'],income)
    dataset['Country'],testset['Country'] = target_encode(dataset['Country'],testset['Country'],income)
    dataset['University Degree'],testset['University Degree'] = target_encode(dataset['University Degree'],testset['University Degree'],income)
    dataset['Profession'],testset['Profession'] = target_encode(dataset['Profession'],testset['Profession'],income)
    dataset['Hair Color'],testset['Hair Color'] = target_encode(dataset['Hair Color'],testset['Hair Color'],income)
    dataset['Housing Situation'],testset['Housing Situation'] = target_encode(dataset['Housing Situation'],testset['Housing Situation'],income)
    dataset['Satisfation with employer'],testset['Satisfation with employer'] = target_encode(dataset['Satisfation with employer'],testset['Satisfation with employer'],income)
    dataset['Year of Record'],testset['Year of Record'] = target_encode(dataset['Year of Record'],testset['Year of Record'],income)
    X=dataset.drop(['Total Yearly Income [EUR]'], axis=1)
    Y=dataset['Total Yearly Income [EUR]']
    xtrain,xvalidate,ytrain,yvalidate=train_test_split(X,Y,test_size=0.2,random_state=0)
    x_test=testset.drop("Total Yearly Income [EUR]",axis=1)
   
   

#Applying model
    regressor=BayesianRidge()
    regressor.fit(xtrain,ytrain)
    y_predict=regressor.predict(xvalidate)
    result=regressor.predict(x_test)
    res=pd.DataFrame(x_test['Instance'])
    res['Total Yearly Income [EUR]']=result
    res.index=x_test.index
    res.to_csv("result.csv")
    rms=np.sqrt(mean_squared_error(yvalidate,y_predict))
    print("rmes is "+ str(rms))


# In[43]:


def run_rmse(dataset,testset):
    print("Preprocessing Dataset and TestSet")
    dataset,testset=preprocessing(dataset,testset)
    #print("One hot encoding")
    #dataset,testset=one_hot_encoding(dataset,testset)
    print("Using predict model")
    model_predict(dataset,testset)
    
if __name__ == '__main__':
    run_rmse(dataset,testset)    


# In[ ]:





# In[ ]:




