PROBLEM STATEMENT:

A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
Which variables are significant in predicting the price of a car?
How well those variables describe the price of a car?
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.


BUSINESS GOAL

We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. 
They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

LIBRARIES USED

Pandas,matplot(visualizing),seaborn(visualizing) for EDA

Prediciton method used here is Multiple Linear Regression

FEATURE ELIMINATION
1)Features that are highly correlated are eliminated using VIF(Variable Inflation Factor) method

2)I've also tried Backward elimination method that eliminated features with p>0.05

FEATURE EXTRACTION
Performed PCA to extract features with high variance from the dataset
  
OPTIMIZATION TECHNIQUES USED
Used ridge and lasso method to add little bit of slop to avoid overfitting.
Also tried out Hyper Parameter tuning to select significant variables

EVALUATION

Evaluated the model using r2 method and got a score of 94%.So, the prediction model is correct 94% of the time.

