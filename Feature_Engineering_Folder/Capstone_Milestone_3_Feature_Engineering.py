import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def load_data():
    # Define the folder where your data is stored
    data_folder = '/workspaces/my-data-product-poisson/Data/'  # Update the path if needed
    
    # Define the path to your data_clean file (adjust if needed)
    data_clean_path = os.path.join(data_folder, 'data_clean.csv')
    
    # Check if the file exists
    if not os.path.exists(data_clean_path):
        print(f"Error: File not found at {data_clean_path}")
        return None

    try:
        # Load the data_clean dataset
        data_clean = pd.read_csv(data_clean_path)
        print("data_clean dataset loaded successfully.")
        
        return data_clean
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Call the load_data function to load data_clean
data_clean = load_data()

# Check if data_clean was loaded successfully
if data_clean is not None:
    print("Data loaded successfully.")
else:
    print("Failed to load data.")


def preprocess_data(data):
    # Add 'day_of_year', 'sin_year', 'cos_year' columns
    data['Day'] = pd.to_datetime(data['Day'])  # Ensure 'Day' is datetime
    data['day_of_year'] = data['Day'].dt.dayofyear
    data['sin_year'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['cos_year'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
    
    return data

def remove_outliers_mean(data, threshold=3):
    relevant_cols = ['COVID-19 doses (daily)', 'Daily new confirmed deaths due to COVID-19']
    for col in relevant_cols:
        if col in data.columns:
            mean = data[col].mean()
            std_dev = data[col].std()
            data = data[~((data[col] - mean).abs() > threshold * std_dev)]
    return data
# removes outliers by standard deviations away
def remove_outliers_median(data, threshold=1.5):
    relevant_cols = ['COVID-19 doses (daily)', 'Daily new confirmed deaths due to COVID-19']
    for col in relevant_cols:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data
# removes outliers by iqrs away
def apply_log_transformation(data):
    relevant_cols = ['COVID-19 doses (daily)', 'Daily new confirmed deaths due to COVID-19']
    for col in relevant_cols:
        if col in data.columns:
            data[col] = np.log(data[col].clip(lower=0.1))  # Avoid log(0) by clipping to 0.1
    return data
# log transforms data

# Assuming 'data_clean' is already loaded
data_clean = preprocess_data(data_clean)

# Create new datasets with the required transformations
data_nomean_out = remove_outliers_mean(data_clean.copy(), threshold=3)
data_nomedian_out = remove_outliers_median(data_clean.copy(), threshold=1.5)
data_log = apply_log_transformation(data_clean.copy())

# Return the datasets for further use in another file
print("Datasets with transformations are ready.")

def prepare_data(data):
    # Select the features and target variable
    X = data[['COVID-19 doses (daily)', 'sin_year', 'cos_year']]  # Features
    y = data['Daily new confirmed deaths due to COVID-19']  # Target variable

    # Split the data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data and transform
    X_test_scaled = scaler.transform(X_test)  # Transform test data using the fitted scaler

    return X_train_scaled, X_test_scaled, y_train, y_test

# Prepare data for each dataset
X_train_clean, X_test_clean, y_train_clean, y_test_clean = prepare_data(data_clean)
X_train_nomean_out, X_test_nomean_out, y_train_nomean_out, y_test_nomean_out = prepare_data(data_nomean_out)
X_train_nomedian_out, X_test_nomedian_out, y_train_nomedian_out, y_test_nomedian_out = prepare_data(data_nomedian_out)
X_train_log, X_test_log, y_train_log, y_test_log = prepare_data(data_log)

def poisson_regression(X_train, y_train, X_test, y_test):
    # Add constant (intercept) term to the features
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit Poisson Regression model
    poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    # Make predictions
    predictions = poisson_model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Poisson Regression MSE: {mse}")
    print(f"Poisson Regression R-squared: {r2}")

    return predictions, mse, r2
# runs poisson regression

def random_forest(X_train, y_train, X_test, y_test):
    # Initialize the Random Forest model
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Make predictions
    predictions = rf.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Random Forest MSE: {mse}")
    print(f"Random Forest R-squared: {r2}")

    return predictions, mse, r2
# runs random forest

def deep_neural_network(X_train, y_train, X_test, y_test, epochs=10):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer + 1st hidden layer
    model.add(Dense(32, activation='relu'))  # 2nd hidden layer
    model.add(Dense(1))  # Output layer (for regression)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    # Make predictions
    predictions = model.predict(X_test).flatten()  # Flatten to get the same shape as y_test

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"DNN MSE: {mse}")
    print(f"DNN R-squared: {r2}")

    return predictions, mse, r2
# runs dnn

def plot_predictions(y_actual, y_pred, feature_name, model_name):
    
    # Create the scatter plot
    fig = px.scatter(
        x=y_actual,
        y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title=f'{model_name} Predictions vs Actuals',
        trendline="ols",  # Optionally include a trendline (ordinary least squares)
        trendline_color_override="red"
    )
    
    # Optionally, add labels or customize the plot further
    fig.update_layout(
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )

    # Show the plot
    fig.show()
    
    return fig
# creates graphs
