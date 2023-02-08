
# Simple Linear Regression Project 


## Modelling the linear relationship between the various features of a medical insurance like age, sex, ibm and children Dataset

&nbsp; &nbsp; &nbsp; &nbsp;


The contents of this project are divided into following topics which are listed as follows:- 




## Table of Contents


1.	Introduction
2.	License information
3.	Python libraries
4.	The problem statement
5.	Linear Regression
6.	Independent and dependent variable
7.	Simple Linear Regression (SLR)
8.	About the dataset
9.	Exploratory data analysis
10.	Mechanics of Simple Linear Regression
12.	Making predictions
13.	Regression metrics for model performance like
 i.     RMSE
ii.      MSE
III.     MAE

14.	Interpretation and conclusion
15.	Residual analysis
16.	Checking for Overfitting or Underfitting the data
17.	Simple Linear Regression - Model Assumptions
18.	References

&nbsp; &nbsp; &nbsp; &nbsp;

## 1.	Introduction

In this project, I build a Simple Linear Regression model to study the linear relationship between Age, sex, region, ibm and the target variable "charges" of the medical insurance dataset. I discuss the basics of linear regression and its implementation in Python programming language using Scikit-learn. Scikit-learn is the popular machine learning library of Python programming language.
 
## 2.	License information

The work done in this Jupyter notebook is made available under the Creative Commons Attribution License 4.0. 

I have licensed this Jupyter notebook for general public. The work done in this project is for learning and demonstration purposes. 
  
&nbsp; &nbsp; &nbsp; &nbsp;


## 3.	Python libraries

I have Anaconda Python distribution installed on my system. It comes with most of the standard Python libraries I need for this project. The basic Python libraries used in this project are:-

 •	Numpy – It provides a fast numerical array structure and operating functions.
 
 •	pandas – It provides tools for data storage, manipulation and analysis tasks.
 
 •	Scikit-Learn – The required machine learning library in Python.
 
 •	Matplotlib – It is the basic plotting library in Python. It provides tools for making plots. 


&nbsp; &nbsp; &nbsp; &nbsp;
	The problem statement

## 4The aim of building a machine learning model is to solve a problem and to define a metric to measure model performance. So, first of all I have to define the problem to be solved in this project.
As described earlier, the problem is to model and investigate the linear relationship between Sales and Advertising dataset for a dietary weight control product. I have used two performance metrics RMSE (Root Mean Square Value) and R2 Score value to compute our model performance.


&nbsp; &nbsp; &nbsp; &nbsp;

## 5.	Linear Regression

Linear Regression is a statistical technique which is used to find the linear relationship between dependent and one or more independent variables. This technique is applicable for Supervised Learning Regression problems where we try to predict a continuous variable.
Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. In this project, I employ Simple Linear Regression technique where I have six independent and one dependent variable. It is the simplest form of Linear Regression where we compare correlation between the variables. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 6.	Independent and Dependent Variables

In this project, I refer Independent variable as Feature variable and Dependent variable as Target variable. These variables are also recognized by different names as follows: -


### Independent variable

Independent variable is also called Input variable and is denoted by X. In practical applications, independent variable is also called Feature variable or Predictor variable. We can denote it as: -
Independent or Input variable (X) = Feature variable = Predictor variable 


### Dependent variable

Dependent variable is also called Output variable and is denoted by y. Dependent variable is also called Target variable or Response variable. It can be denote it as follows: -

Dependent or Output variable (y) = Target variable = Response variable


&nbsp; &nbsp; &nbsp; &nbsp;

## 7.	Linear Regression 

Linear Regression is the simplest model in machine learning. It models the linear relationship between the independent and dependent variables. 

In this project, there are six independent or input variable which represents the data and is denoted by X. Similarly, there is one dependent or output variable which represents the y value and is denoted by y. We want to build a linear relationship between these variables. This  relationship can be modelled by mathematical equation of the form:-

				Y = β0   + β1*X    

In this equation, X and Y are called independent and dependent variables respectively, 
β1 is the coefficient for independent variable and
β0 is the constant term.
β0 and β1 are called parameters of the model.
 
For simplicity, we can compare the above equation with the basic line equation of the form:-
 
        			y = ax + b       ----------------- (2)
        

We can see that 


slope of the line is given by, a =  β1,  and

intercept of the line by b =  β0. 

In this Simple Linear Regression model, we want to fit a line which estimates the linear relationship between X and Y. So, the question of fitting reduces to estimating the parameters of the model β0 and β1. 

 
### Ordinary Least Square Method (OLS)


&nbsp; &nbsp; &nbsp; &nbsp;



 
Now, our task is to find a line which best fits the above scatter plot. This line will help us to predict the value of any Target variable for any given Feature variable. This line is called regression line. 
We can define an error function for any line. Then, the regression line is the one which minimizes the error function. Such an error function is also called a Costor a Cost function. 

&nbsp; &nbsp; &nbsp; &nbsp;


### Cost Function



We want the above line to resemble the dataset as closely as possible. In other words, we want the line to be as close to actual data points as possible. It can be achieved by minimizing the vertical distance between the actual data point and fitted line. We calculate the vertical distance between each data point and the line. This distance is called the residual. So, in a regression model, we try to minimize the residuals by finding the line of best fit. 


&nbsp; &nbsp; &nbsp; &nbsp;

Diagrammatic representation of residuals is given below. In this diagram, the residuals are represented by the vertical dotted lines from actual data points to the line.


&nbsp; &nbsp; &nbsp; &nbsp;

![](Images/Diagrammatic%20Representation%20of%20Residuals.png)


We can try to minimize the sum of the residuals, but then a large positive residual would cancel out a large negative residual. For this reason, we minimize the sum of the squares of the residuals. 

Mathematically, we denote actual data points by yi and predicted data points by ŷi. So, the residual for a data point i would be given as 
				
       				
				di = yi -  ŷi
				

Sum of the squares of the residuals is given as:


				 D = Ʃ di2       for all data points

This is the Cost function. It denotes the total error present in the model which is the sum of the total errors of each individual data point. We can represent it diagrammatically as follows:-

&nbsp; &nbsp; &nbsp; &nbsp;


![](Images/Ordinary%20Least%20Squares.png)


We can estimate the parameters of the model β0 and β1 by minimize the error in the model by minimizing D. Thus, we can find the regression line given by equation (1).  This method of finding the parameters of the model and thus regression line is called Ordinary Least Square Method.

&nbsp; &nbsp; &nbsp; &nbsp;


## 8.	About the dataset

This data set contains sex, age, children,ibm and region columns. It contains medical insurance data. 

&nbsp; &nbsp; &nbsp; &nbsp;

