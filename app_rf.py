import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, log_step  # Import log_step

# Load your data files
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# Detailed comment to be printed in the browser
detailed_comment = """
(EXAMPLE- WILL CHANGE DEPENDENT ON ITERATION) Random Forest modeling did show more promise in its results. The model based on the original data produced a higher R-squared value of -0.18012783493528484. However, the mean squared error was still high, at 48,222,758,103,529.266. Cross validation showed great differences in MSE and R-squared values. The observed against predicted values had some values gathered around the diagonal through the graph, but more were distributed away from the diagonal. The model based on removing outliers three standard deviations away resulted in a lower mean squared error, (14,747,701,344,638.822), but it was still large, and the R-squared improved to -0.3859638774809353. Cross validation results were more consistent. The observed against predicted values graph did have values more disperse than the original, but still few were situated around a diagonal. As for the data set with the outliers removed using the interquartile range, the mean squared error was large, but still better than the original data's, at 9,303,333,520,386.098. It's R-squared value was -6.108212974529689, which suggests an error with the model as R-squared values should be between -1 and 1. Interestingly, cross validation showed other models having R-squared values ranging from -0.277 to -0.295, and MSEs similar to that of the model created. It is possible that the created model is an outlier. Additionally, its predicted against observed values were mostly situated more towards the bottom of the graph. The model generated from the logarithmically transformed data received the best overall results, with a mean squared error of 49.05173430731382 and an R-squared value of -0.13586517945585053. Cross validation did have similar MSEs and R-squared values. While there was still variation in the data points, the values observed against predicted graph do appear to be better distributed along the forty-five-degree center of the graph. Further modifications to this model could result in greater performance. Overall, the logarithmic models appear to be the best way to handle outliers in this large data set. However, across various modeling, Poisson Regression, Random Forest, and DNN models had similar results with logarithmic transformed data. Unfortunately, no models generated results strong enough to make reliable decisions upon. Given this, the next steps for this research is to implement feature engineering techniques to attempt to improve model performance. If this does not improve the results, then local data may be more appropriate when creating predictive models. The former will be performed in a new file, as well as the later if poor results are received again. Insights could potentially be gained on predicting the vaccination needs for future purposes, as well as providing information to policy makers on potential effective techniques in limiting the deaths due to the virus.
"""

# Layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': 'Clean Data', 'value': 'clean'},
                 {'label': 'No Mean Outliers', 'value': 'nomean_out'},
                 {'label': 'No Median Outliers', 'value': 'nomedian_out'},
                 {'label': 'Log Data', 'value': 'log'}],
        value='clean',  # default value
        style={'width': '50%'}
    ),
    dcc.Graph(id='graph'),
    html.Div(id='performance-metrics'),  # Div to display performance metrics
    html.Div(id='log-output'),  # Div to display logs
    html.Div(id='comment-output', children=detailed_comment)  # Div to display detailed comment
])

@app.callback(
    [dash.dependencies.Output('graph', 'figure'),
     dash.dependencies.Output('performance-metrics', 'children'),
     dash.dependencies.Output('log-output', 'children')],  # Output for log messages
    [dash.dependencies.Input('dataset-dropdown', 'value')]
)
def update_graph(selected_dataset):
    # Log the dataset selection step
    log_message = log_step(f"Selected dataset: {selected_dataset}")

    # Choose the appropriate data based on selection
    if selected_dataset == "clean":
        X = data_clean["COVID-19 doses (daily)"]
        y = data_clean["Daily new confirmed deaths due to COVID-19"]
    elif selected_dataset == "nomean_out":
        X = data_nomean_out["COVID-19 doses (daily, no outliers)"]
        y = data_nomean_out["Daily new confirmed deaths due to COVID-19 (no outliers)"]
    elif selected_dataset == "nomedian_out":
        X = data_nomedian_out["COVID-19 doses (daily, no outliers)"]
        y = data_nomedian_out["Daily new confirmed deaths due to COVID-19 (no outliers)"]
    elif selected_dataset == "log":
        X = data_log["COVID-19 doses (daily)"]
        y = data_log["Daily new confirmed deaths due to COVID-19"]

    # Reshape X for use with sklearn (if it's a 1D array)
    X = X.values.reshape(-1, 1)

    # Initialize Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

    try:
        # Log cross-validation step
        log_step("Performing cross-validation for R-squared and MSE.")

        # Perform cross-validation for R-squared and MSE
        r2_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')  # R-squared
        mse_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')  # MSE

        # Fit the model
        rf_model.fit(X, y)

        # Predictions for the model
        y_pred = rf_model.predict(X)

        # MSE for the model
        mse = mean_squared_error(y, y_pred)

        # Log performance metrics
        log_step(f"Model performance: MSE = {mse:.4f}, R-squared = {rf_model.score(X, y):.4f}")
        log_step(f"Cross-validation R-squared: Mean = {r2_scores.mean():.4f}, Std = {r2_scores.std():.4f}")
        log_step(f"Cross-validation MSE: Mean = {-mse_scores.mean():.4f}, Std = {mse_scores.std():.4f}")

    except Exception as e:
        log_step(f"Error during model fitting or cross-validation: {e}")
        return None, "Error occurred while processing the data.", "Error occurred while processing the data."

    # Prepare performance metrics text
    performance_text = (
        f"Mean Squared Error (MSE): {mse:.4f}<br>"
        f"R-squared (train set): {rf_model.score(X, y):.4f}<br>"
        f"Cross-validation R-squared: Mean = {r2_scores.mean():.4f}, Std = {r2_scores.std():.4f}<br>"
        f"Cross-validation MSE: Mean = {-mse_scores.mean():.4f}, Std = {mse_scores.std():.4f}"
    )

    # Create a scatter plot with predictions
    fig = px.scatter(
        x=X.flatten(),  # Flatten X for plotting
        y=y,
        title=f"COVID-19 Doses vs. Deaths: {selected_dataset.capitalize()} Data",
        trendline="ols",  # Adding line of best fit (OLS)
        trendline_color_override="red"  # Set the trendline color to red
    )

    # adds performance metrics to graph, so it can be downoloaded with the graph as a png file
    fig.add_annotation(
    x=0.5, y=0.95,  
    text=performance_text,
    showarrow=False,
    font=dict(size=12, color="black"),
    align="center",
    bgcolor="white",
    borderpad=4,
    )

    # Return the graph, performance metrics, and log message
    return fig, performance_text, log_message

# Run the server
if __name__ == '__main__':
    log_step("Starting the Dash app.")  # This line is valid in this scope
    app.run(debug=True)  # Run the app
    log_step("Dash app is running.")  # This will only run after the app starts

