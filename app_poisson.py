import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, train_poisson_model, evaluate_model, visualize_predictions, log_step  # Import log_step

# Load your data files
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# Detailed comment to be printed in the browser
detailed_comment = """
(EXAMPLE- WILL CHANGE DEPENDENT ON ITERATION) The Poisson Regression models did not produce optimistic results. In the original data, the R-squared value returned a score of 0.002693973499347746, 
and the mean standard error was found as 40,839,273,939,600.09. Predicted against observed values produced a graph with few data points lying near the perfect prediction line. 
Cross validation showed a wide array of responses, but all MSE's were high and R-squared were low, indicating consistent poor fit. The results of the data set with the outliers removed 
given the standard deviations were a mean standard error of 10,631,726,924,267.488 and a R-squared value of 0.0019319613551788128. The data did seem to migrate closer to the diagonal line 
than the original data. Cross validation showed a consistent MSE, but inconsistent R-squared. It's overall performance improved from the prior model, but is still not a good model. 
As for those with outliers handled through interquartile range, the mean standard error was found to be 1,308,816,023,268.6018 and the R-squared was -0.0000010858840857608243. 
Again, MSE was consistent, but R-squared was not. The graph of original against predicted values reflected this, with a horizontal line. This all still shows a poorly fit model. 
As for the logarithmic transformed results, the scores did improve. The R-squared value was 0.09858806028453393, and the mean squared error lowered to 36.940747655014526. 
Cross validation results were more consistent than all other models. The observed against predicted values graph did have significantly more data points fall on the perfect prediction line, 
but there were still many scattered away from it. The logarithmic transformed data is the best model created through Poisson Regression. The Poisson Regression at this point in the 
process did not produce a viable prediction graph. Implementing feature engineering techniques may result in a better performance, but other modeling techniques may still show superior results.

Overall, the logarithmic models appear to be the best way to handle outliers in this large data set. However, across various modeling, Poisson Regression, Random Forest, 
and DNN models had similar results with logarithmic transformed data. Unfortunately, no models generated results strong enough to make reliable decisions upon. 
Given this, the next steps for this research is to implement feature engineering techniques to attempt to improve model performance. If this does not improve the results, 
then local data may be more appropriate when creating predictive models. The former will be performed in a new file, as well as the later if poor results are received again. 
Insights could potentially be gained on predicting the vaccination needs for future purposes, as well as providing information to policy makers on potential effective techniques in limiting the deaths due to the virus.
"""

# Layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Clean Data', 'value': 'clean'},
            {'label': 'No Mean Outliers', 'value': 'nomean_out'},
            {'label': 'No Median Outliers', 'value': 'nomedian_out'},
            {'label': 'Log Data', 'value': 'log'}
        ],
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

    # Train-test split for cross-validation (not strictly needed for this, but just for clarity)
    model = LinearRegression()

    try:
        # Log cross-validation step
        log_step("Performing cross-validation for R-squared and MSE.")

        # Perform cross-validation for R-squared and MSE
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')  # R-squared
        mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # MSE

        # Fit the model
        model.fit(X, y)

        # Predictions for trendline (for plotting)
        y_pred = model.predict(X)

        # MSE for the model
        mse = mean_squared_error(y, y_pred)

        # Log performance metrics
        log_step(f"Model performance: MSE = {mse:.4f}, R-squared = {model.score(X, y):.4f}")
        log_step(f"Cross-validation R-squared: Mean = {r2_scores.mean():.4f}, Std = {r2_scores.std():.4f}")
        log_step(f"Cross-validation MSE: Mean = {-mse_scores.mean():.4f}, Std = {mse_scores.std():.4f}")

    except Exception as e:
        log_step(f"Error during model fitting or cross-validation: {e}")
        return None, "Error occurred while processing the data.", "Error occurred while processing the data."

    # Prepare performance metrics text
    performance_text = (
        f"Mean Squared Error (MSE): {mse:.4f}<br>"
        f"R-squared (train set): {model.score(X, y):.4f}<br>"
        f"Cross-validation R-squared: Mean = {r2_scores.mean():.4f}, Std = {r2_scores.std():.4f}<br>"
        f"Cross-validation MSE: Mean = {-mse_scores.mean():.4f}, Std = {mse_scores.std():.4f}"
    )

    # Create a scatter plot with a red line of best fit (OLS)
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
    log_step("Starting the Dash app.")
    app.run(debug=True, port = 7000)
    log_step("Dash app is running.")

