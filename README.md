# Regression Analysis Using Linear regression and Boosting Algorithms

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Theroretical Aspect](#theoretical-aspect)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Credits](#credits)


![](https://www.bankforeclosuressale.com/images/categories/cream-multi-family-homes.jpg)

## Overview
This model is used to predict sale prize of a House on the basis of different aspects like Neighborhood, Street, Roof Material etc. Kaggle Dataset contains 1460 rows and 81 columns. 43 categorical columns, 37 numerical columns and 1 target column(SalePrize). Target column is continuous in nature so Regression Analysis is used in all algorithms. In Data Exploration univariant alaysis and bivariant analysis is done, along with it Outliers are detected and removed using Robust Scaling. Firstly, Datset is trained on Ridge Linear Regression and then Bossting algorithms( Gradiant Boosting, Light GBM and XGBM)  are used. After comparing R2 score and Mean Squared error of all the algorithms, Model with Light GBM algorithm is selected. 


## Motivation
It is the first model that I build after learning Linear Regression and Data Exploration as a course assignment. Then after I learned about outlier dectection and  Boosting algorithms and thought of modifying my assignment. 

## Technical Aspect
1. The entire project is completed on Google Colab, which provide a Jupyter notebook environment that runs entirely in the cloud.
2. For visualization Matplot, Seaborn, Plotly libraries are used.
3. For statistics Scipy.stats is used.
4. For Feature Engineering 
     * Sklearn.preprocessing is used
     * Category_encoders is used
5. For Model Building
     * Sklearn.linear_models for Linear Regression
     * Sklearn.ensembles for Gradiant Boosting algorithm
     * Light GBM and XGB algorithm from Sklearn
6. For Model Evaluation sklearn.metrics is used

## Theoretical Aspect
### Exploratory Data Analysis
   1. Univariant Analysis  
      * Numerical columns Histogram is used.
      * Categorical columns Barplot is used.
   2. Bivariant Analysis 
      * Continuous Numerical - Target column Scatter plot and Correlation matrix is used.
      * Discrete Numerical - Target column Violin plot is used.
      * Categorical - Target column Barplot is used
### Outliers Detection
   For outliers detection IQR method is used.


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```


## To Do
1. Using other different ML algorithms to find the accuracy of the prediction.
2. Hyperparameter Tuning can be done to improve accuracy of the existing model.

## Bug / Feature Request
If you find a bug (the model couldn't handle the query and / or gave undesired results), kindly open an issue [<a href = "https://github.com/itspb008/Used-Car-Quality-Detection/issues/new">here</a>] by including your search query and the expected result.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.software.ac.uk/sites/default/files/images/content/jupyter-main-logo.svg" width=300>](https://jupyter.org/)    [<img target="_blank" src = "https://www.bgp4.com/wp-content/uploads/2019/08/Scikit_learn_logo_small.svg_-840x452.png" width=170>](https://scikit-learn.org/stable/)                     [<img target="_blank" src="https://miro.medium.com/max/3880/1*ddtqWGkJz1TUCg1WM9qKeQ.jpeg" width=280>](https://colab.research.google.com/) 



[<img target="_blank" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" width=200>](https://seaborn.pydata.org/)                  [<img target="_blank" src="https://blueorange.digital/wp-content/uploads/2019/12/logo_matplotlib.jpg" width=200>](https://matplotlib.org/stable/index.html) 


## Team
Prashant Bharti


## Credits
- https://jovian.ai/learn/machine-learning-with-python-zero-to-gbms
- Machine Learning Course by Andrew NG on Coursera
- Analytics Vidhya Blogs 
- Krish Naik Youtube Channel
- StatQuest with Josh Stramer Youtube Channel
- Kaggle.com
