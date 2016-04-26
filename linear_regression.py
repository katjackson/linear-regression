import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

"""Ground Cricket Chirps"""
# import grasshopper data
ground_cricket_data = {"Chirps/Second": [20.0, 16.0, 19.8, 18.4, 17.1, 15.5,
                                         14.7, 15.7, 15.4, 16.3, 15.0, 17.2,
                                         16.0, 17.0, 14.4],
                       "Ground Temperature": [88.6, 71.6, 93.3, 84.3, 80.6,
                                              75.2, 69.7, 71.6, 69.4, 83.3,
                                              79.6, 82.6, 80.6, 83.5, 76.3]}
df = pd.DataFrame(ground_cricket_data)

# instantiate LinearRegression class
regr = linear_model.LinearRegression()

# define variables
x = df['Ground Temperature']
# pandas.Series.to_frame() returns a data frame
x = x.to_frame()
y = df['Chirps/Second']

# fit the object to the data
regr.fit(x, y)

# plot using equation
plt.scatter(x, y)
# use attributes to plot linear regression equation
# y = β0 + β1x where β0 is intercept and β1 is coefficient
plt.plot(x, (regr.intercept_ + (regr.coef_ * x)))

# plot using max/min as data for prediction
# create a new data frame of min and max feature values
df_new = pd.DataFrame({'Ground Temp': [df['Ground Temperature'].min(),
                       df['Ground Temperature'].max()]})
plt.scatter(x, y)
# plot the new frame against the prediction
plt.plot(df_new, regr.predict(df_new))

# calculate the r-squared score (or coefficient of determination)
regr.score(x, y)

# predict a response based on a hypothetical feature 95
regr.predict(95)

# predict the feature based on a hypothetical response 18
# used algebra to solve for x in y = regr.intercept_ + (regr.coef_ * x)
(18 - regr.intercept_) / regr.coef_

"""Brain vs. Body Weight"""
# import data
bb = pd.read_fwf('brain_body.txt')

# instantiate LinearRegression class
bb_regr = linear_model.LinearRegression()

# define variables
X = bb['Brain'].to_frame()
y = bb['Body']

# fit the object to the data
bb_regr.fit(X, y)

# plot using equation
plt.scatter(X, y, s=10)
# use attributes to plot linear regression equation
# y = β0 + β1x where β0 is intercept and β1 is coefficient
plt.plot(X, (bb_regr.intercept_ + (bb_regr.coef_ * X)))

# calculate the r-squared score (or coefficient of determination)
bb_regr.score(X, y)

"""Salary Discrimination"""
# import salary data
sd = pd.read_fwf("salary.txt", header=None,
                 names=["Sex", "Rank", "Year", "Degree", "YSdeg", "Salary"])

# Instantiate the LinearRegression class
sd_regr = linear_model.LinearRegression()

# set variables including multiple features
feature_cols = ['Sex', 'Rank', 'Year', 'Degree', 'YSdeg']
X = sd[feature_cols]
y = sd['Salary']

# fit linear model to data
sd_regr.fit(X, y)

# find elements fo linear regression equation
print(sd_regr.intercept_, sd_regr.coef_)

# Multiple Linear Regression Equation
# y = β0 + β1x +...+βnxn

# find r2 score for each feature by running new linear regression
for x in feature_cols:
    x_regr = linear_model.LinearRegression()
    x_regr.fit(sd[x].to_frame(), sd['Salary'])
    r2 = x_regr.score(sd[x].to_frame(), sd['Salary'])
    print('r2 for {} = {}'.format(x, r2))

# plot each feature against the salary repsonse with linear regression
for x in feature_cols:
    x_regr = linear_model.LinearRegression()
    x_regr.fit(sd[x].to_frame(), sd['Salary'])

    plt.scatter(sd[x], sd['Salary'])
    plt.plot(sd[x], (x_regr.intercept_ + (x_regr.coef_ * sd[x])))

    plt.title("{} vs. Salary".format(x))
    plt.show()

# plot all of the data and the linear regression model
# looks wierd and confusing
df_new = pd.DataFrame({'Sex': [sd['Sex'].min(), sd['Sex'].max()],
                       'Rank': [sd['Rank'].min(), sd['Rank'].max()],
                       'Year': [sd['Year'].min(), sd['Year'].max()],
                       'Degree': [sd['Degree'].min(), sd['Degree'].max()],
                       'YSDeg': [sd['YSdeg'].min(), sd['YSdeg'].max()]})

for x in feature_cols:
    plt.scatter(sd[x].to_frame(), y)
plt.plot(df_new, sd_regr.predict(df_new))
plt.show()

# make 3D plot that doesn't prove anything
input_data = sd[['Sex', 'Rank']]
output_data = sd['Salary']

b_regr = linear_model.LinearRegression()
b_regr.fit(input_data, sd['Salary'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx = input_data['Sex']
yy = input_data['Rank']
zz = output_data

predict = b_regr.predict(input_data)
x_surf, y_surf = np.meshgrid(xx, yy)

ax.plot_surface(x_surf, y_surf, predict, color='red', alpha=0.1)
ax.scatter(xx, yy, zz)

plt.xlabel('Sex')
plt.ylabel('Rank')
