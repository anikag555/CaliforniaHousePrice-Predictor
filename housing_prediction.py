
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Loads the training data
housing_data = pd.read_csv('housing.csv')

# Display first few rows to verify data is loading
print(housing_data.head())

# Separate features and target variable
x = housing_data.drop('MedHouseVal', axis = 1)
y = housing_data['MedHouseVal']

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, 
    test_size = 0.2, random_state = 42)

# Intialize the model
model = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Predict on the valdiation set
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_val)

# Make predictions
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {mae}')