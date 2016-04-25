
# import grasshopper data
ground_cricket_data = {"Chirps/Second": [20.0, 16.0, 19.8, 18.4, 17.1, 15.5, 14.7,
                                         15.7, 15.4, 16.3, 15.0, 17.2, 16.0, 17.0,
                                         14.4],
                       "Ground Temperature": [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7,
                                              71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5,
                                              76.3]}
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
df_new = pd.DataFrame({'Ground Temp': [df['Ground Temperature'].min(), df['Ground Temperature'].max()]})
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
