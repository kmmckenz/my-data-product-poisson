import numpy as np
import plotly.graph_objs as go
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
from Feature_Engineering_Folder.Capstone_Milestone_3_Feature_Engineering import (
    load_data, preprocess_data, remove_outliers_mean, remove_outliers_median, apply_log_transformation,
    poisson_regression, random_forest, deep_neural_network, plot_predictions
)
from sklearn.model_selection import train_test_split

# Initialize the Dash app
app = Dash(__name__)

# Load and preprocess the data
data_clean = load_data()
data_clean = preprocess_data(data_clean)

# Create new datasets based on transformations
data_nomean_out = remove_outliers_mean(data_clean.copy(), threshold=3)
data_nomedian_out = remove_outliers_median(data_clean.copy(), threshold=1.5)
data_log = apply_log_transformation(data_clean.copy())

# Split the data into features and target
def split_data(data):
    X = data[['COVID-19 doses (daily)', 'sin_year', 'cos_year']]  # Features
    y = data['Daily new confirmed deaths due to COVID-19']  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Clip extreme values for the predictions and actual values
def plot_predictions(y_actual, y_pred, feature_name, model_name):
    # Clip the values to a range that is more manageable for plotting
    y_actual = np.clip(y_actual, -1000, 1000)  # Adjust this range as necessary
    y_pred = np.clip(y_pred, -1000, 1000)

    fig = go.Figure(data=[
        go.Scatter(
            x=y_actual, 
            y=y_pred, 
            mode='markers', 
            marker=dict(color='blue', size=5),
            name='Predictions vs Actuals'
        )
    ])

    # Add trendline (linear regression line)
    fig.update_traces(marker=dict(color='blue'), line=dict(dash='solid', color='red'))
    
    fig.update_layout(
        title=f'{model_name} Predictions vs Actuals',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )

    return fig

# Define the layout for the Dash app
app.layout = html.Div([
    html.H1("Model Performance Metrics"),

    # Dropdown to select the dataset
    html.Label("Select Dataset:"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Original Data (data_clean)', 'value': 'data_clean'},
            {'label': 'Data with Mean Outliers Removed (data_nomean_out)', 'value': 'data_nomean_out'},
            {'label': 'Data with Median Outliers Removed (data_nomedian_out)', 'value': 'data_nomedian_out'},
            {'label': 'Log Transformed Data (data_log)', 'value': 'data_log'}
        ],
        value='data_clean',  # default value
    ),

    # Dropdown to select the model
    html.Label("Select Model:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Poisson Regression', 'value': 'poisson'},
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'Deep Neural Network', 'value': 'dnn'}
        ],
        value='poisson',  # default value
    ),
    
    # Placeholder for the output
    html.Div(id='model-output'),
    dcc.Graph(id='model-graph')
])

# Callback to update the output based on dataset and model selected
@app.callback(
    [Output('model-output', 'children'),
     Output('model-graph', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_output(dataset, model):
    # Select the dataset based on the user's selection
    if dataset == 'data_clean':
        data = data_clean
    elif dataset == 'data_nomean_out':
        data = data_nomean_out
    elif dataset == 'data_nomedian_out':
        data = data_nomedian_out
    elif dataset == 'data_log':
        data = data_log

    # Reduce the data size for debugging purposes
    data_subset = data.sample(1000)  # Limit to 1000 rows for testing

    # Split the selected data into features and target
    X_train, X_test, y_train, y_test = split_data(data_subset)

    # Run the selected model
    if model == 'poisson':
        predictions, mse, r2 = poisson_regression(X_train, y_train, X_test, y_test)
        model_name = 'Poisson Regression'
    elif model == 'rf':
        predictions, mse, r2 = random_forest(X_train, y_train, X_test, y_test)
        model_name = 'Random Forest'
    elif model == 'dnn':
        predictions, mse, r2 = deep_neural_network(X_train, y_train, X_test, y_test)
        model_name = 'Deep Neural Network'

    # Create the prediction graph with clipped values to improve rendering
    fig = plot_predictions(y_test, predictions, 'COVID-19 doses (daily)', model_name)

    # Display the model's performance metrics
    metrics = [
        html.P(f"MSE: {mse:.2f}"),
        html.P(f"R²: {r2:.2f}")
    ]\

    return metrics, fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port= 9555)

