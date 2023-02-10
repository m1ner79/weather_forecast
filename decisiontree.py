#Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score, cross_validate, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# Load data
data = pd.read_csv('/content/drive/MyDrive/Y4 Data Science/CSV files/merged-files-2-mod.csv')

# A correlation heatmap used to determine the correlation between different features
plt.figure(figsize=(10,8))
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Split the date column into day, month and year
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Create feature and target variables
X = data[['latitude',	'longitude', 'day', 'month', 'year', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl']]
y = data['temp']

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

# Baseline model using mean value of temperature

# Mean of the target variable (temperature) in the training set
baseline = np.mean(y_train)

# Predicted temperature values for the baseline model
y_pred_baseline = [baseline for i in range(len(y_test))]

# The mean squared error (MSE) is calculated as the average of the squared differences between the actual and predicted temperature values
baseline_mse = mean_squared_error(y_test, y_pred_baseline)

# The R^2 score is calculated as the ratio of the explained variance to the total variance in the target variable
# It is a measure of the goodness of fit of a model
baseline_r2 = r2_score(y_test, y_pred_baseline)

# he mean absolute error (MAE) is calculated as the average of the absolute differences between the actual and predicted temperature values.
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

# The lower the value, the better the model's accuracy in predicting the temperature.
print("Baseline MAE: {:.5f}".format(baseline_mae))

# The higher the value, the better the model fits the data.
print("Baseline R^2: {:.5f}".format(baseline_r2))

#  The lower the value, the better the model's accuracy in predicting the temperature.
print("Baseline MSE: {:.5f}".format(baseline_mse))

# Decision tree model

# The code creates a decision tree regressor object using the DecisionTreeRegressor()
dt_reg = DecisionTreeRegressor()

# The fit() function is then used to train the model on the training data, where X_train is the independent variable and y_train is the dependent variable.
dt_reg.fit(X_train, y_train)

# The predicted values for the dependent variable
y_pred_dt = dt_reg.predict(X_test)

# The next three lines calculate performance metrics for the decision tree model using 
# the actual temperature values from the test set and the predicted temperature values from the decision tree model.
dt_mse = mean_squared_error(y_test, y_pred_dt)
dt_mae = mean_absolute_error(y_test, y_pred_dt)
dt_r2 = r2_score(y_test, y_pred_dt)

# The lower the value, the better the model's accuracy in predicting the temperature.
print("Decision Tree MAE: {:.5f}".format(dt_mae))

# The higher the value, the better the model fits the data.
print("Decision Tree R^2: {:.5f}".format(dt_r2))

#  The lower the value, the better the model's accuracy in predicting the temperature.
print("Decision Tree MSE: {:.5f}".format(dt_mse))

# Linear regression model

# The code creates a decision tree regressor object using the LinearRegression()
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_mae = mean_absolute_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)
print("Linear regression MAE: {:.5f}".format(lin_mae))
print("Linear regression R^2: {:.5f}".format(lin_r2))
print("Linear regression MSE: {:.5f}".format(lin_mse))

# Random forest model

rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
print("Random forest MAE: {:.5f}".format(rf_mae))
print("Random forest R^2: {:.5f}".format(rf_r2))
print("Random forest MSE: {:.5f}".format(rf_mse))

# Evaluate the models
print("Baseline Model - MSE: {:.5f}".format(baseline_mse), " R2 Score: {:.5f}".format(baseline_r2))
print("Decision Tree Model - MSE: {:.5f}".format(dt_mse), " R2 Score: {:.5f}".format(dt_r2))
print("Linear Regression Model - MSE: {:.5f}".format(lin_mse), " R2 Score: {:.5f}".format(lin_r2))
print("Random Forest Model - MSE: {:.5f}".format(rf_mse), " R2 Score: {:.5f}".format(rf_r2))

# MSE - The lower the value, the better the model's accuracy in predicting the temperature.
# R2 - # The higher the value, the better the model fits the data, with 1 being a perfect fit and 0 being a poor fit

# Plot the comparison of predictions for each model
plt.figure(figsize=(10,8))
plt.scatter(y_test, y_pred_dt, label='Decision Tree')
plt.scatter(y_test, y_pred_lin, label='Linear Regression')
plt.scatter(y_test, y_pred_rf, label='Random Forest')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

# Select the best model and make predictions on the test data
model = None
if dt_mae < lin_mae and dt_mae < rf_mae:
    model = dt_reg
elif lin_mae < dt_mae and lin_mae < rf_mae:
    model = lin_reg
else:
    model = rf_reg

print("Best Model to make predictions: ", model)

#Interpreting the model and reporting results visually and numerically

#Get feature importance from the Random Forest model
if model == rf_reg:
    feature_importance = rf_reg.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

#Evaluate the selected model's performance on the test set and compare it to the baseline model
test_predictions = model.predict(X_test)
test_mae = mean_absolute_error(test_predictions, y_test)
print("Selected Model Test MAE: {:.5f}".format(test_mae))
print("Baseline Test MAE: {:.5f}".format(baseline_mae))

#Plotting actual vs predicted values for test data
plt.figure(figsize=(10,8))
plt.scatter(y_test, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# This is k-fold cross validation. The data is divided into k folds, and the model is trained 
# on k-1 folds and tested on the remaining fold. This process is repeated k times, 
# with each fold being used as the test set once. The average performance of the model on 
# all k folds is used as an estimate of the model's performance on unseen data.

# Perform cross-validation

#  The first argument is the regression model to be evaluated, 
# and the second and third arguments are the X and y variables used for training the model. 
# The "cv" argument specifies the number of folds for the cross-validation. 
# The "return_train_score" argument is set to True, which means that the performance of the model
# on the training set will be returned along with the performance on the validation set.
dt_reg_cv = cross_validate(dt_reg, X_train, y_train, cv=5, return_train_score=True)
lin_reg_cv = cross_validate(lin_reg, X_train, y_train, cv=5, return_train_score=True)
rf_reg_cv = cross_validate(rf_reg, X_train, y_train, cv=5, return_train_score=True)

# Calculate the mean and standard deviation of the cross-validation scores

# The mean (average) is a measure of the center of the data, calculated by summing all the values
# and dividing by the number of values. The standard deviation is a measure of the spread of the data
# and indicates how much the data deviates from the mean.
dt_mean = np.mean(dt_reg_cv['test_score'])
dt_std = np.std(dt_reg_cv['test_score'])
lr_mean = np.mean(lin_reg_cv['test_score'])
lr_std = np.std(lin_reg_cv['test_score'])
rf_mean = np.mean(rf_reg_cv['test_score'])
rf_std = np.std(rf_reg_cv['test_score'])

print("Decision Tree cross-validation mean: {:.2f} +/- {:.2f}".format(dt_mean, dt_std))
print("Linear Regression cross-validation mean: {:.2f} +/- {:.2f}".format(lr_mean, lr_std))
print("Random Forest cross-validation mean: {:.2f} +/- {:.2f}".format(rf_mean, rf_std))

plt.figure(figsize=(10, 8))
# Plot the predicted vs actual temperatures for the decision tree model
plt.scatter(y_test, y_pred_dt, color='red', label='Decision Tree')

# Plot the predicted vs actual temperatures for the linear regression model
plt.scatter(y_test, y_pred_lin, color='blue', label='Linear Regression')

# Plot the predicted vs actual temperatures for the random forest model
plt.scatter(y_test, y_pred_rf, color='green', label='Random Forest')

# Plot a line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Add labels and title to the plot
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.title('Predicted vs actual temperature')
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(10, 8))
# Plot the predicted vs actual temperatures for the decision tree model
plt.scatter(y_test, y_pred_dt, color='orange', label='Decision Tree')

# Plot a line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Add labels and title to the plot
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.title('Predicted vs actual temperature')
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(10, 8))
# Plot the predicted vs actual temperatures for the linear regression model
plt.scatter(y_test, y_pred_lin, color='orange', label='Linear Regression')

# Plot a line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Add labels and title to the plot
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.title('Predicted vs actual temperature')
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(10, 8))
# Plot the predicted vs actual temperatures for the random forest model
plt.scatter(y_test, y_pred_rf, color='orange', label='Random Forest')

# Plot a line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Add labels and title to the plot
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.title('Predicted vs actual temperature')
plt.legend()

# Show the plot
plt.show()

# Predict Decision Tree on training set
dt_train_predictions = dt_reg.predict(X_train)
train_mse = mean_squared_error(y_train, dt_train_predictions)
train_rmse = np.sqrt(train_mse)
print('Decision Tree Training Set MSE: {:.5f}'.format(train_mse))
print('Decision Tree Training Set RMSE: {:.5f}'.format(train_rmse))

# Predict Decision Treeon validation set
dt_validation_predictions = dt_reg.predict(X_val)
validation_mse = mean_squared_error(y_val, dt_validation_predictions)
validation_rmse = np.sqrt(validation_mse)
print('Decision Tree Validation Set MSE: {:.5f}'.format(validation_mse))
print('Decision Tree Validation Set RMSE: {:.5f}'.format(validation_rmse))

# Predict Decision Tree on test set
dt_test_predictions = dt_reg.predict(X_test)
test_mse = mean_squared_error(y_test, dt_test_predictions)
test_rmse = np.sqrt(test_mse)
print('Decision Tree Test Set MSE: {:.5f}'.format(test_mse))
print('Decision Tree Test Set RMSE: {:.5f}'.format(test_rmse))

# Plot the comparison of MSE and RMSE
train_errors = [train_mse, validation_mse, test_mse]
rmse_errors = [train_rmse, validation_rmse, test_rmse]
x = ['Training Set', 'Validation Set', 'Test Set']
plt.plot(x, train_errors, label='MSE')
plt.plot(x, rmse_errors, label='RMSE')
plt.legend()
plt.xlabel('Data Sets')
plt.ylabel('Error')
plt.title('Comparison of MSE and RMSE of Decision Tree on Data Sets')
plt.show()

# Predict Linear Regression on training set
lin_train_predictions = lin_reg.predict(X_train)
train_mse = mean_squared_error(y_train, lin_train_predictions)
train_rmse = np.sqrt(train_mse)
print('Linear Regression Training Set MSE: {:.5f}'.format(train_mse))
print('Linear Regression Training Set RMSE: {:.5f}'.format(train_rmse))

# Predict Linear Regression validation set
lin_validation_predictions = lin_reg.predict(X_val)
validation_mse = mean_squared_error(y_val, lin_validation_predictions)
validation_rmse = np.sqrt(validation_mse)
print('Linear Regression Validation Set MSE: {:.5f}'.format(validation_mse))
print('Linear Regression Validation Set RMSE: {:.5f}'.format(validation_rmse))

# Predict Linear Regression on test set
lin_test_predictions = lin_reg.predict(X_test)
test_mse = mean_squared_error(y_test, lin_test_predictions)
test_rmse = np.sqrt(test_mse)
print('Linear Regression Test Set MSE: {:.5f}'.format(test_mse))
print('Linear Regression Test Set RMSE: {:.5f}'.format(test_rmse))

# Plot the comparison of MSE and RMSE
train_errors = [train_mse, validation_mse, test_mse]
rmse_errors = [train_rmse, validation_rmse, test_rmse]
x = ['Training Set', 'Validation Set', 'Test Set']
plt.plot(x, train_errors, label='MSE')
plt.plot(x, rmse_errors, label='RMSE')
plt.legend()
plt.xlabel('Data Sets')
plt.ylabel('Error')
plt.title('Comparison of MSE and RMSE of Linear Regression on Data Sets')
plt.show()

# Predict Random Forest on training set
rf_train_predictions = rf_reg.predict(X_train)
train_mse = mean_squared_error(y_train, rf_train_predictions)
train_rmse = np.sqrt(train_mse)
print('Random Forest Training Set MSE: {:.5f}'.format(train_mse))
print('Random Forest Training Set RMSE: {:.5f}'.format(train_rmse))

# Predict Random Forest validation set
rf_validation_predictions = rf_reg.predict(X_val)
validation_mse = mean_squared_error(y_val, rf_validation_predictions)
validation_rmse = np.sqrt(validation_mse)
print('Random Forest Validation Set MSE: {:.5f}'.format(validation_mse))
print('Random Forest Validation Set RMSE: {:.5f}'.format(validation_rmse))

# Predict Random Forest on test set
rf_test_predictions = rf_reg.predict(X_test)
test_mse = mean_squared_error(y_test, rf_test_predictions)
test_rmse = np.sqrt(test_mse)
print('Random Forest Test Set MSE: {:.5f}'.format(test_mse))
print('Random Forest Test Set RMSE: {:.5f}'.format(test_rmse))

# Plot the comparison of MSE and RMSE
train_errors = [train_mse, validation_mse, test_mse]
rmse_errors = [train_rmse, validation_rmse, test_rmse]
x = ['Training Set', 'Validation Set', 'Test Set']
plt.plot(x, train_errors, label='MSE')
plt.plot(x, rmse_errors, label='RMSE')
plt.legend()
plt.xlabel('Data Sets')
plt.ylabel('Error')
plt.title('Comparison of MSE and RMSE of Random Forrest on Data Sets')
plt.show()

# Initialize the ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Initialize a list to store the scores and errors for each iteration
dt_scores = []
dt_mse = []
dt_rmse = []

# Decision Tree cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    dt_reg.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = dt_reg.predict(X_test)
    
    # Evaluate the model's performance
    score = dt_reg.score(X_test, y_test)
    dt_scores.append(score)
   
    mse = mean_squared_error(y_test, y_pred)
    dt_mse.append(mse)
    rmse = np.sqrt(mse)
    dt_rmse.append(rmse)

    print(f'Decision Tree Score - {score}, MSE - {mse:.5f}, RMSE - {rmse:.5f}')

# This bar plot will show the score for each iteration of the shuffle split. 
# The score is a measure of the model's performance on the test data for each iteration.
# A higher score indicates a better fit, with 1 being a perfect fit.

# Plot the scores

plt.bar(range(len(dt_scores)), dt_scores)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Decision Tree Shuffle Split Scores')
plt.show()

# Initialize a list to store the scores for each iteration
lin_scores = []
lin_mse = []
lin_rmse = []

# Linear Regression cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    lin_reg.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = lin_reg.predict(X_test)
    
    # Evaluate the model's performance
    score = lin_reg.score(X_test, y_test)
    lin_scores.append(score)
    
    
    mse = mean_squared_error(y_test, y_pred)
    lin_mse.append(mse)
    rmse = np.sqrt(mse)
    lin_rmse.append(rmse)

    print(f'Linear Regression Score - {score}, MSE - {mse:.5f}, RMSE - {rmse:.5f}')

# Plot the scores
plt.bar(range(len(lin_scores)), lin_scores)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Linear Regression Shuffle Split Scores')
plt.show()

# Initialize a list to store the scores for each iteration
rf_scores = []
rf_mse = []
rf_rmse = []

# Random Forrest cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    rf_reg.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = rf_reg.predict(X_test)
    
    # Evaluate the model's performance
    score = rf_reg.score(X_test, y_test)
    rf_scores.append(score)
  
    mse = mean_squared_error(y_test, y_pred)
    rf_mse.append(mse)
    rmse = np.sqrt(mse)
    rf_rmse.append(rmse)

    print(f'Random Forrest Score - {score}, MSE - {mse:.5f}, RMSE - {rmse:.5f}')

# Plot the scores
plt.bar(range(len(rf_scores)), rf_scores)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Random Forest Shuffle Split Scores')
plt.show()

# Initialize the ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Initialize a list to store the scores for each iteration
dt_scores = []
train_rmse_list = []
val_rmse_list = []

# Decision Tree cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    dt_reg.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = dt_reg.predict(X_train)

    # Make predictions on the validation data
    y_val_pred = dt_reg.predict(X_test)

    # Evaluate the model's performance on the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_rmse_list.append(train_rmse)

    # Evaluate the model's performance on the validation set
    val_mse = mean_squared_error(y_test, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_rmse_list.append(val_rmse)

    # Evaluate the model's performance on the test set
    test_mse = mean_squared_error(y_test, y_val_pred)
    test_rmse = np.sqrt(test_mse)

    # Add the score to the list of scores
    dt_scores.append(score)

# Plot the performance of the model on the training, validation, and test sets
plt.figure(figsize=(10, 8))
plt.plot(train_rmse_list, label='Training Set')
plt.plot(val_rmse_list, label='Validation Set')
plt.plot([test_rmse for i in range(len(train_rmse_list))], label='Test Set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Decision Tree Model Performance')
plt.show()

# Initialize the ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Initialize a list to store the scores for each iteration
lin_scores = []
train_rmse_list = []
val_rmse_list = []

# Linear Regression cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    lin_reg.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = lin_reg.predict(X_train)

    # Make predictions on the validation data
    y_val_pred = lin_reg.predict(X_test)

    # Evaluate the model's performance on the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_rmse_list.append(train_rmse)

    # Evaluate the model's performance on the validation set
    val_mse = mean_squared_error(y_test, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_rmse_list.append(val_rmse)

    # Evaluate the model's performance on the test set
    test_mse = mean_squared_error(y_test, y_val_pred)
    test_rmse = np.sqrt(test_mse)

    # Add the score to the list of scores
    lin_scores.append(score)

# Plot the performance of the model on the training, validation, and test sets
plt.figure(figsize=(10, 8))
plt.plot(train_rmse_list, label='Training Set')
plt.plot(val_rmse_list, label='Validation Set')
plt.plot([test_rmse for i in range(len(train_rmse_list))], label='Test Set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Linear Regression Model Performance')
plt.show()

# Initialize the ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Initialize a list to store the scores for each iteration
rf_scores = []
train_rmse_list = []
val_rmse_list = []

# Random Forest cross-validation
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training data
    rf_reg.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = rf_reg.predict(X_train)

    # Make predictions on the validation data
    y_val_pred = rf_reg.predict(X_test)

    # Evaluate the model's performance on the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_rmse_list.append(train_rmse)

    # Evaluate the model's performance on the validation set
    val_mse = mean_squared_error(y_test, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_rmse_list.append(val_rmse)

    # Evaluate the model's performance on the test set
    test_mse = mean_squared_error(y_test, y_val_pred)
    test_rmse = np.sqrt(test_mse)

    # Add the score to the list of scores
    rf_scores.append(score)

# Plot the performance of the model on the training, validation, and test sets
plt.figure(figsize=(10, 8))
plt.plot(train_rmse_list, label='Training Set')
plt.plot(val_rmse_list, label='Validation Set')
plt.plot([test_rmse for i in range(len(train_rmse_list))], label='Test Set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Random Forest Model Performance')
plt.show()
