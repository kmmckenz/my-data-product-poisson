import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
from Poisson_Folder.Capstone_Milestone_3_Poisson import load_data, log_step  # Import log_step

# Load your data files
data_clean, data_nomean_out, data_nomedian_out, data_log = load_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# Detailed comment to be printed in the browser
detailed_comment = """
(EXAMPLE- WILL CHANGE DEPENDENT ON ITERATION) The Deep Neural Network models did not provide compelling evidence towards effectively learning the data. As seen in the prior models, all but the logarithmic transformed data performed poorly. The original data saw a mean squared error of 43,642,823,956,364.0 and an R-squared score of -0.06804573963836003. The observed against predicted values showed data as a horizontal line to the center of the graph. With outliers removed based on the standard deviations, the mean squared error was 12,532,594,920,075.143 and the R-squared was -0.177791944955514454. The scatterplot generated from the observed against predicted values showed values dispersed all over the graph, with the majority located at the edges. The model removing outliers through the interquartile range showed a minor improvement, with the mean squared error at 1,583,783,899,306.115 and the R-squared score at -0.21009025821012806. The graph showed the values to be even more diversly spread all throughout the area of the graph. Finally, the logarithmic data had a mean squared error of 36.59449774004095 and an R-squared of 0.1526004874329061. More data\
was centralized along the diagonal area of the graph, but it still showed great variation. Feature engineering techniques could be explored to improve the Deep Neural Network models. Overall, the logarithmic models appear to be the best way to handle outliers in this large data set. However, across various modeling, Poisson Regression, Random Forest, and DNN models had similar results with logarithmic transformed data. Unfortunately, no models generated results strong enough to make reliable decisions upon. Given this, the next steps for this research is to implement feature engineering techniques to attempt to improve model performance. If this does not improve the results, then local data may be more appropriate when creating predictive models. The former will be performed in a new file, as well as the later if poor results are received again. Insights could potentially be gained on predicting the vaccination needs for future purposes, as well as providing information to policy makers on potential effective techniques in limiting the deaths due to the virus.
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and compile the DNN model
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam())

    try:
        # Log model training step
        log_step("Training DNN model.")

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate MSE and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log performance metrics
        log_step(f"Model performance: MSE = {mse:.4f}, R-squared = {r2:.4f}")

    except Exception as e:
        log_step(f"Error during model training: {e}")
        return None, "Error occurred while processing the data.", "Error occurred while processing the data."

    # Prepare performance metrics text
    performance_text = (
        f"Mean Squared Error (MSE): {mse:.4f}<br>"
        f"R-squared (test set): {r2:.4f}<br>"
    )

    # Create a scatter plot with predictions
    fig = px.scatter(
        x=X_test.flatten(),  # Flatten X for plotting
        y=y_test,
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
    app.run(debug=True)
    log_step("Dash app is running.")
