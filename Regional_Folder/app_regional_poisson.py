import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import sys
import os

# Ensure access to Poisson module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, log_step

# Load data
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  

detailed_comment = """
(EXAMPLE- WILL CHANGE DEPENDENT ON ITERATION) The Poisson Regression model with data from the United States and Canada did not provide compelling evidence towards effectively learning the data. The United States showed a MSE of 1.0748 and R-Squared of 0.0101, with inconsistent R-squared results in the cross validation. As for Canada, a R-squared value of 0.0785 was found, and a mean squared error 1.2258. Cross validation again showed inconsistencies in the R-squared value. Additionally, while the mean squared error is small, R-squared values are too low. Given these results, the model is not recommended to make inferences from.
"""

# App layout
app.layout = html.Div([
    html.H1("COVID-19 Deaths vs Vaccinations: Poisson Regression"),
    
    html.Label("Select Dataset"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Log Data', 'value': 'clean'},
            # You can add more datasets here
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
            usa_data = data_clean[data_clean["Entity"] == "United States"].copy()
            usa_data = usa_data[
                (usa_data['COVID-19 doses (daily)'] > 0) &
                (usa_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            usa_data['COVID-19 doses (daily)'] = np.log(usa_data['COVID-19 doses (daily)'])
            usa_data['Daily new confirmed deaths due to COVID-19'] = np.log(usa_data['Daily new confirmed deaths due to COVID-19'])
            X_us = usa_data["COVID-19 doses (daily)"].values.reshape(-1, 1)
            y_us = usa_data["Daily new confirmed deaths due to COVID-19"].values

            can_data = data_clean[data_clean["Entity"] == "Canada"].copy()
            can_data = can_data[
                (can_data['COVID-19 doses (daily)'] > 0) &
                (can_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            can_data['COVID-19 doses (daily)'] = np.log(can_data['COVID-19 doses (daily)'])
            can_data['Daily new confirmed deaths due to COVID-19'] = np.log(can_data['Daily new confirmed deaths due to COVID-19'])
            X_can = can_data["COVID-19 doses (daily)"].values.reshape(-1, 1)
            y_can = can_data["Daily new confirmed deaths due to COVID-19"].values

            model_us = LinearRegression()
            model_can = LinearRegression()

            r2_scores_us = cross_val_score(model_us, X_us, y_us, cv=5, scoring='r2')
            mse_scores_us = cross_val_score(model_us, X_us, y_us, cv=5, scoring='neg_mean_squared_error')
            r2_scores_can = cross_val_score(model_can, X_can, y_can, cv=5, scoring='r2')
            mse_scores_can = cross_val_score(model_can, X_can, y_can, cv=5, scoring='neg_mean_squared_error')

            model_us.fit(X_us, y_us)
            model_can.fit(X_can, y_can)

            y_pred_us = model_us.predict(X_us)
            y_pred_can = model_can.predict(X_can)

            mse_us = mean_squared_error(y_us, y_pred_us)
            mse_can = mean_squared_error(y_can, y_pred_can)

            log_messages = [
                log_step(f"US Model MSE = {mse_us:.4f}, R² = {model_us.score(X_us, y_us):.4f}"),
                log_step(f"US CV R²: Mean = {r2_scores_us.mean():.4f}, Std = {r2_scores_us.std():.4f}"),
                log_step(f"US CV MSE: Mean = {-mse_scores_us.mean():.4f}, Std = {mse_scores_us.std():.4f}"),
                log_step(f"Canada Model MSE = {mse_can:.4f}, R² = {model_can.score(X_can, y_can):.4f}"),
                log_step(f"Canada CV R²: Mean = {r2_scores_can.mean():.4f}, Std = {r2_scores_can.std():.4f}"),
                log_step(f"Canada CV MSE: Mean = {-mse_scores_can.mean():.4f}, Std = {mse_scores_can.std():.4f}")
            ]

            fig = px.scatter()
            fig.add_scatter(x=X_us.flatten(), y=y_us, mode='markers', name='US Actual')
            fig.add_scatter(x=X_can.flatten(), y=y_can, mode='markers', name='Canada Actual')

            performance_text = (
                f"US MSE: {mse_us:.4f}\n"
                f"US R²: {model_us.score(X_us, y_us):.4f}\n"
                f"US CV R²: {r2_scores_us.mean():.4f} ± {r2_scores_us.std():.4f}\n"
                f"US CV MSE: {-mse_scores_us.mean():.4f} ± {mse_scores_us.std():.4f}\n\n"
                f"Canada MSE: {mse_can:.4f}\n"
                f"Canada R²: {model_can.score(X_can, y_can):.4f}\n"
                f"Canada CV R²: {r2_scores_can.mean():.4f} ± {r2_scores_can.std():.4f}\n"
                f"Canada CV MSE: {-mse_scores_can.mean():.4f} ± {mse_scores_can.std():.4f}"
            )

            return fig, performance_text, "\n".join(log_messages)

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        return px.scatter(title="Error"), error_msg, error_msg

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
