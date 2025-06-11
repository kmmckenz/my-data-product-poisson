import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# Ensure access to Poisson module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, log_step

# Load data
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Needed for deployment

detailed_comment = """
(EXAMPLE- WILL CHANGE DEPENDENT ON ITERATION) The DNN model with data from the United States and Canada did not provide compelling evidence towards effectively learning the data. The United States showed a MSE of 1.1971 and R-Squared of -0.1026. As for Canada, a R-squared value of 0.0676 was found, and a mean squared error 1.0470. While the mean squared error is small, R-squared values are too low. Given these results, the model is not recommended to make inferences from.
"""

# App layout
app.layout = html.Div([
    html.H1("COVID-19 Deaths vs Vaccinations: Deep Neural Network Regression"),
    
    html.Label("Select Dataset"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Log Data', 'value': 'clean'},
        ],
        value='clean'
    ),
    
    dcc.Graph(id='graph'),
    
    html.Div(id='performance-metrics', style={'whiteSpace': 'pre-line', 'marginTop': '20px'}),
    
    html.Div(id='log-output', style={'whiteSpace': 'pre-line', 'marginTop': '20px'}),

    html.Div(id='comment-output', children=detailed_comment),
])

# Callback
@app.callback(
    [Output('graph', 'figure'),
     Output('performance-metrics', 'children'),
     Output('log-output', 'children')],
    [Input('dataset-dropdown', 'value')]
)
def update_graph(selected_dataset):
    log_message = log_step(f"Selected dataset: {selected_dataset}")

    try:
        if selected_dataset == "clean":
            # US data
            usa_data = data_clean[data_clean["Entity"] == "United States"].copy()
            usa_data = usa_data[
                (usa_data['COVID-19 doses (daily)'] > 0) & 
                (usa_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            usa_data['COVID-19 doses (daily)'] = np.log(usa_data['COVID-19 doses (daily)'])
            usa_data['Daily new confirmed deaths due to COVID-19'] = np.log(usa_data['Daily new confirmed deaths due to COVID-19'])
            X_us = usa_data["COVID-19 doses (daily)"].values.reshape(-1, 1)
            y_us = usa_data["Daily new confirmed deaths due to COVID-19"].values

            # Canada data
            can_data = data_clean[data_clean["Entity"] == "Canada"].copy()
            can_data = can_data[
                (can_data['COVID-19 doses (daily)'] > 0) & 
                (can_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            can_data['COVID-19 doses (daily)'] = np.log(can_data['COVID-19 doses (daily)'])
            can_data['Daily new confirmed deaths due to COVID-19'] = np.log(can_data['Daily new confirmed deaths due to COVID-19'])
            X_can = can_data["COVID-19 doses (daily)"].values.reshape(-1, 1)
            y_can = can_data["Daily new confirmed deaths due to COVID-19"].values

            # Define DNN model
            def create_model():
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(265, activation='relu', input_shape=(1,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                return model

            # Train model for US
            model_us = create_model()
            model_us.fit(X_us, y_us, epochs=100, verbose=0)
            y_pred_us = model_us.predict(X_us).flatten()

            # Train model for Canada
            model_can = create_model()
            model_can.fit(X_can, y_can, epochs=100, verbose=0)
            y_pred_can = model_can.predict(X_can).flatten()

            # Metrics
            mse_us = mean_squared_error(y_us, y_pred_us)
            r2_us = r2_score(y_us, y_pred_us)
            mse_can = mean_squared_error(y_can, y_pred_can)
            r2_can = r2_score(y_can, y_pred_can)

            log_messages = [
                log_step(f"US Model MSE = {mse_us:.4f}, R² = {r2_us:.4f}"),
                log_step(f"Canada Model MSE = {mse_can:.4f}, R² = {r2_can:.4f}")
            ]

            # Visualization
            fig = px.scatter()
            fig.add_scatter(x=X_us.flatten(), y=y_us, mode='markers', name='US Actual')
            fig.add_scatter(x=X_can.flatten(), y=y_can, mode='markers', name='Canada Actual')

            performance_text = (
                f"US MSE: {mse_us:.4f}\n"
                f"US R²: {r2_us:.4f}\n\n"
                f"Canada MSE: {mse_can:.4f}\n"
                f"Canada R²: {r2_can:.4f}"
            )

            return fig, performance_text, "\n".join(log_messages)

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        return px.scatter(title="Error"), error_msg, error_msg

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

