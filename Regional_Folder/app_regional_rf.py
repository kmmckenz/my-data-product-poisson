import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import os
import sys

# ---- Safe Import Handling for Local Module ----
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in some environments Jupyter
    current_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, log_step

# ---- Load Preprocessed Data ----
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# ---- Dash App Initialization ----
app = dash.Dash(__name__)
server = app.server  # For deployment

detailed_comment = """
(EXAMPLE - WILL CHANGE DEPENDENT ON ITERATION)
The Random Forest model with data from the United States and Canada did provide the best performance over all other models. The United States showed a MSE of 0.1980 and R-Squared of 0.8176. Cross validation showed average R-squared scores at -1.4249 and mean squared error at 1.6988. As for Canada, a R-squared value of 0.8372 was found, and a mean squared error 0.1828. Cross validation showed R-squared values averaging at -1.2408 and mean squared errors at 1.6008. Given the cross validation R-squared scores, it is still advised to not use this model for any influential decision making.
"""

# ---- Layout ----
app.layout = html.Div([
    html.H1("COVID-19 Deaths vs Vaccinations: Random Forest Regression"),

    html.Label("Select Dataset"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': 'Log Data', 'value': 'clean'}],
        value='clean'
    ),

    dcc.Graph(id='graph'),

    html.Div(id='performance-metrics', style={'whiteSpace': 'pre-line', 'marginTop': '20px'}),
    html.Div(id='log-output', style={'whiteSpace': 'pre-line', 'marginTop': '20px'}),
    html.Div(id='comment-output', children=detailed_comment)
])

# ---- Callback ----
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
            # --- Preprocess US Data ---
            usa_data = data_clean[data_clean["Entity"] == "United States"].copy()
            usa_data = usa_data[
                (usa_data['COVID-19 doses (daily)'] > 0) &
                (usa_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            usa_data['COVID-19 doses (daily)'] = np.log(usa_data['COVID-19 doses (daily)'])
            usa_data['Daily new confirmed deaths due to COVID-19'] = np.log(usa_data['Daily new confirmed deaths due to COVID-19'])
            X_us = usa_data[["COVID-19 doses (daily)"]].values
            y_us = usa_data["Daily new confirmed deaths due to COVID-19"].values

            # --- Preprocess Canada Data ---
            can_data = data_clean[data_clean["Entity"] == "Canada"].copy()
            can_data = can_data[
                (can_data['COVID-19 doses (daily)'] > 0) &
                (can_data['Daily new confirmed deaths due to COVID-19'] > 0)
            ]
            can_data['COVID-19 doses (daily)'] = np.log(can_data['COVID-19 doses (daily)'])
            can_data['Daily new confirmed deaths due to COVID-19'] = np.log(can_data['Daily new confirmed deaths due to COVID-19'])
            X_can = can_data[["COVID-19 doses (daily)"]].values
            y_can = can_data["Daily new confirmed deaths due to COVID-19"].values

            # --- Models ---
            model_us = RandomForestRegressor(n_estimators=100, random_state=42)
            model_can = RandomForestRegressor(n_estimators=100, random_state=42)

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

            # --- Graph ---
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

# ---- Run App ----
if __name__ == '__main__':
    app.run(debug=True)

