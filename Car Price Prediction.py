# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:13 2020

@author: Chandramouli
"""

import pandas as pd

dataset=pd.read_csv("CarPrice.csv")
y=dataset.iloc[:,-1]
x=dataset.iloc[:,:-1]
x=x.drop(['car_ID','symboling'],axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset.info()
dataset.shape

#EDA
total = x.isnull().sum().sort_values(ascending=False)#to find no of null values in each  column
total1=x.isnull().count().sort_values(ascending=False)
x.info()
percent = x.isnull().sum()/x.isnull().count().sort_values(
        ascending=False)*100
        
missing_data = pd.concat([total,percent], axis=1,
                         keys=['Total','Percent'])
#no missing values in any of the columns
x.dtypes
numerics = ['int64', 'float64']
cat_num=x.select_dtypes(include=numerics)
#outlier analysis
plt.figure(figsize=(14, 6))
plt.subplot(2,3,1)
sns.boxplot(x = 'wheelbase', data = cat_num)
plt.subplot(2,3,2)
sns.boxplot(x = 'carlength', data = cat_num)
plt.subplot(2,3,3)
sns.boxplot(x = 'carwidth', data = cat_num)
plt.subplot(2,3,4)
sns.boxplot(x = 'carheight', data = cat_num)
plt.subplot(2,3,5)
sns.boxplot(x = 'curbweight', data = cat_num)
plt.subplot(2,3,6)
sns.boxplot(x = 'enginesize', data = cat_num)
plt.figure(figsize=(15, 6))
plt.subplot(2,3,1)
sns.boxplot(x = 'boreratio', data = cat_num)
plt.subplot(2,3,2)
sns.boxplot(x = 'stroke', data = cat_num)
plt.subplot(2,3,3)
sns.boxplot(x = 'compressionratio', data = cat_num)
plt.subplot(2,3,4)
sns.boxplot(x = 'horsepower', data = cat_num)
plt.subplot(2,3,5)
sns.boxplot(x = 'peakrpm', data = cat_num)
plt.subplot(2,3,6)
sns.boxplot(x = 'citympg', data = cat_num)
plt.figure(figsize=(15, 3))
plt.subplot(1,2,1)
sns.boxplot(x = 'highwaympg', data = cat_num)


#sns.distplot(cat_num['wheelbase'])#to check distribution

cat_features= x.select_dtypes(include='object')
def plotgraph(cat_features,colname):
    sns.countplot(x=colname,data=cat_features)
    plt.title(colname)
    plt.show()

plt.subplot(2,2,1)
plotgraph(cat_features,'enginelocation')
plt.subplot(2,2,2)
plotgraph(cat_features,'fueltype')
plt.subplot(2,2,3)
plotgraph(cat_features,'aspiration')
plt.subplot(2,2,4)
plotgraph(cat_features,'doornumber')
plt.subplot(2,3,5)
plotgraph(cat_features,'carbody')
plt.subplot(2,3,6)
plotgraph(cat_features,'drivewheel')
plt.show()

#inferences
#The engine is mostly located in the front of the car
#Most of the cars use gas as their fuel
#The aspiration employed by most vehicles is std (standard)
#Just over half the cars sold have four doors
#The most popular car body is sedan
#Most of the cars have a fwd drive 

###
x['CarName'] = x['CarName'].str.split('-').str[0]#to get the brand names
x['CarName'] = x['CarName'].str.split(' ').str[0]#to get the brand names

x['CarName'] = x['CarName'].str.lower()

x['CarName'] = x['CarName'].str.replace('vw','volkswagen')
x['CarName'] = x['CarName'].str.replace('maxda','mazda')
x['CarName'] = x['CarName'].str.replace('vokswagen','volkswagen')
x['CarName'] = x['CarName'].str.replace('toyouta','toyota')

x['drivewheel'] = x['drivewheel'].str.replace('4wd','fwd')

cat_features['CarName'] = cat_features['CarName'].str.split('-').str[0]#to get the brand names
cat_features['CarName'] = cat_features['CarName'].str.split(' ').str[0]#to get the brand names

cat_features['CarName'] = cat_features['CarName'].str.lower()

cat_features['CarName'] = cat_features['CarName'].str.replace('vw','volkswagen')
cat_features['CarName'] = cat_features['CarName'].str.replace('maxda','mazda')
cat_features['CarName'] = cat_features['CarName'].str.replace('vokswagen','volkswagen')
cat_features['CarName'] = cat_features['CarName'].str.replace('toyouta','toyota')

cat_features['drivewheel'] = cat_features['drivewheel'].str.replace('4wd','fwd')

x_cat=pd.get_dummies(cat_features)
#dropping columns 
x_cat.info()
x_cat=x_cat.drop(['CarName_alfa' ,'fueltype_diesel','aspiration_std','doornumber_four','carbody_convertible','drivewheel_rwd','enginelocation_front','enginetype_dohc','cylindernumber_eight','fuelsystem_1bbl'],axis=1)

x1=pd.concat([x_cat,cat_num],axis=1)
#scaling values feature scaling should not be done in regression because regression coefficients will compensate to put everything under same scale
#from sklearn.preprocessing import StandardScaler
#scale=StandardScaler()
#x2=scale.fit_transform(cat_num)
#x2=pd.DataFrame(x2)
#cat_num.columns
#x2.columns=cat_num.columns
#x3=pd.concat([x_cat,x2],axis=1)
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x1,y,test_size = 0.3, random_state = 100)


#Correlation #VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor    
def vif_calc(X):
  import numpy as np
  thresh=3
  cols = x_train.columns
  variables = np.arange(x_train.shape[1])
  dropped=True
  while dropped:
       dropped=False
       c = x_train[cols[variables]].values
       vif = [variance_inflation_factor(c, i) for i in np.arange(c.shape[1])]
       maxloc = vif.index(max(vif))
       if max(vif) > thresh:
        print('dropping \'' + x_train[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
        variables = np.delete(variables, maxloc)
        dropped=True
  print('Remaining variables:')
  print(x_train.columns[variables])
  return x_train[cols[variables]]
vif=vif_calc(x_train)
x_train1=vif
set(x_test)-set(x_train1)#to find out unique columns in train compared to test

x_test1=x_test.drop(['CarName_peugeot',
 'CarName_porsche',
 'CarName_subaru',
 'boreratio',
 'carbody_hatchback',
 'carbody_sedan',
 'carheight',
 'carlength',
 'carwidth',
 'citympg',
 'compressionratio',
 'curbweight',
 'cylindernumber_five',
 'cylindernumber_four',
 'cylindernumber_six',
 'cylindernumber_two',
 'drivewheel_fwd',
 'enginesize',
 'enginetype_ohc',
 'enginetype_rotor',
 'fuelsystem_2bbl',
 'fuelsystem_mpfi',
 'fueltype_gas',
 'highwaympg',
 'horsepower',
 'peakrpm',
 'stroke',
 'wheelbase'],axis=1)
    
#now we have train and test set after VIF calculation
#prediction
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train1,y_train)#gives line of best fit
regression.coef_
regression.intercept_

y_pred=regression.predict(x_test1)
# Plotting y_test and y_pred to understand the spread
plt.scatter(y_test,y_pred)

#
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
#without scaling the r2 score is high
#RIDGE AND LASSO-Optimization techniques for regression models to reduce overfitting
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[0,1e-10,1e-15,1,2,5,10,50,100]}#1e-10=0.000000001#for hyperparmeter tuning
ridge_grid=GridSearchCV(ridge,param_grid=parameters,scoring='r2',cv=10)
ridge_grid.fit(x_train,y_train)
ridge_grid.best_params_#{'alpha': 10}
ridge_grid.best_score_#0.83

#lasso
from sklearn.linear_model import Lasso
lasso=Lasso()
from sklearn.model_selection import GridSearchCV

parameters={'alpha':[0,1e-10,1e-15,1,2,5,10,50,100]}#1e-10=0.000000001#for hyperparmeter tuning
lasso_grid=GridSearchCV(lasso,param_grid=parameters,scoring='r2',cv=10)
lasso_grid.fit(x_train,y_train)
lasso_grid.best_params_
lasso_grid.best_score_#0.81

#PCA
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x2=scale.fit_transform(cat_num)
x2=pd.DataFrame(x2)
cat_num.columns
x2.columns=cat_num.columns
x3=pd.concat([x_cat,x2],axis=1)
#train test split
from sklearn.model_selection import train_test_split
x_train_pca, x_test_pca,y_train_pca,y_test_pca = train_test_split(x3,y,test_size = 0.3, random_state = 100)


from sklearn.decomposition import PCA
pca=PCA()#pca(n_compo=5) will take top 5 feature out of 13 if nothing is given it will take all 13

x_train_pca=pca.fit_transform(x_train)#fing eigen values,eigen vectors and sort it by descending order
x_test_pca=pca.transform(x_test)
explained_ratio=pca.explained_variance_ratio_#features(columns) is in order of descending sort

import numpy as np
np.cumsum(explained_ratio)#cumulative sum out of 13 features up 9 features its 95 percent so we can remove other 4
#but column name is not orderly hence we dont know what can be removed so this is PCA

####
from sklearn.decomposition import PCA
pca=PCA(n_components=10)#pca(n_compo=10) will take top 10 feature out of 63 since they contribute more

x_train_pca=pca.fit_transform(x_train)#find eigen values,eigen vectors and sort it by descending order
x_test_pca=pca.transform(x_test)
explained_ratio=pca.explained_variance_ratio_#features(columns) is in order of descending sort

import numpy as np
np.cumsum(explained_ratio)


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train_pca,y_train_pca)#gives line of best fit
regression.coef_
regression.intercept_

y_pred_pca=regression.predict(x_test_pca)


from sklearn.metrics import r2_score

r2_pca= r2_score(y_test_pca, y_pred_pca)

#OLS
#backward elimination
import statsmodels.api as sm #for stats model always we need to give intercept#for sk learn no need to give intercept
x1['intercept']=1
x1_ols=x1
reg_OLS=sm.OLS(y,x1_ols).fit()#ordinary least square
reg_OLS.summary()
#dropping CarName_audi  p>0.05
x1_ols=x1_ols.drop(['CarName_audi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()#ordinary least square
reg_OLS.summary()
#dropping citympg  p>0.05
x1_ols=x1_ols.drop(['citympg'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping horsepower p>0.05
x1_ols=x1_ols.drop(['horsepower'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_mfi p>0.05
x1_ols=x1_ols.drop(['fuelsystem_mfi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_porcshce p>0.05
x1_ols=x1_ols.drop(['CarName_porcshce'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_4bbl p>0.05
x1_ols=x1_ols.drop(['fuelsystem_4bbl'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_saab p>0.05
x1_ols=x1_ols.drop(['CarName_saab'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_idi p>0.05
x1_ols=x1_ols.drop(['fuelsystem_idi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping enginetype_l p>0.05
x1_ols=x1_ols.drop(['enginetype_l'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping drivewheel_fwd p>0.05
x1_ols=x1_ols.drop(['drivewheel_fwd'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_spdi p>0.05
x1_ols=x1_ols.drop(['fuelsystem_spdi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_spfi p>0.05
x1_ols=x1_ols.drop(['fuelsystem_spfi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping doornumber_two p>0.05
x1_ols=x1_ols.drop(['doornumber_two'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_jaguar p>0.05
x1_ols=x1_ols.drop(['CarName_jaguar'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping cylindernumber_three p>0.05
x1_ols=x1_ols.drop(['cylindernumber_three'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping cylindernumber_two p>0.05
x1_ols=x1_ols.drop(['cylindernumber_two'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping enginetype_rotor p>0.05
x1_ols=x1_ols.drop(['enginetype_rotor'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping enginetype_ohc p>0.05
x1_ols=x1_ols.drop(['enginetype_ohc'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping cylindernumber_twelve p>0.05
x1_ols=x1_ols.drop(['cylindernumber_twelve'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping stroke p>0.05
x1_ols=x1_ols.drop(['stroke'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping carbody_hardtop p>0.05
x1_ols=x1_ols.drop(['carbody_hardtop'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping carbody_sedan p>0.05
x1_ols=x1_ols.drop(['carbody_sedan'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping carbody_wagon p>0.05
x1_ols=x1_ols.drop(['carbody_wagon'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping highwaympg p>0.05
x1_ols=x1_ols.drop(['highwaympg'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping enginetype_dohcv p>0.05
x1_ols=x1_ols.drop(['enginetype_dohcv'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_porsche p>0.05
x1_ols=x1_ols.drop(['CarName_porsche'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping aspiration_turbo p>0.05
x1_ols=x1_ols.drop(['aspiration_turbo'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_mpfi p>0.05
x1_ols=x1_ols.drop(['fuelsystem_mpfi'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_mercury p>0.05
x1_ols=x1_ols.drop(['CarName_mercury'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping CarName_volvo p>0.05
x1_ols=x1_ols.drop(['CarName_volvo'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping wheelbase p>0.05
x1_ols=x1_ols.drop(['wheelbase'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()
#dropping fuelsystem_2bbl p>0.05
x1_ols=x1_ols.drop(['fuelsystem_2bbl'],axis=1)
reg_OLS=sm.OLS(y,x1_ols).fit()
reg_OLS.summary()

# all the features are now below p>0.05
r2_ols=reg_OLS.rsquared#0.957
adj_r2_ols=reg_OLS.rsquared_adj#0.950

from sklearn.model_selection import train_test_split
x_train_ols, x_test_ols,y_train_ols,y_test_ols = train_test_split(x1_ols,y,test_size = 0.3, random_state = 100)
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train_ols,y_train_ols)#gives line of best fit
regression.coef_
regression.intercept_

y_pred_ols=regression.predict(x_test_ols)

from sklearn.metrics import r2_score

r2_ols = r2_score(y_test_ols, y_pred_ols)#0.93
plt.scatter(y_test_ols,y_pred_ols)