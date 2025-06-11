from flask import Flask, render_template_string, request, send_file
from EDA_Folder.Capstone_Milestone_3_EDA import load_data, log_step, log_content, data_describe, about_eda
from EDA_Folder.Capstone_Milestone_3_EDA import creating_interactive_graphs1
from io import BytesIO
import weasyprint

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def eda():
    log_step("Starting EDA")  # Logs EDA start

    data = load_data()  # Load data
    if data is None:
        log_step("Failed to load data")
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to load data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{{ log_content | join('\n') }}</pre>  <!-- Display logs using Jinja filter -->
        """)

    data_clean, data_nomean_out, data_nomedian_out, data_log = data  # Unpack data

    # Log descriptive statistics
    data_describe(data_clean, data_nomean_out, data_nomedian_out, data_log)

    # Get unique countries for the dropdown filter
    unique_countries = data_clean["Entity"].unique()

    # Handle form submission to filter by country
    selected_country = request.form.get('country')  # Selected country from form

    # Generate the interactive graphs by calling the parent function
    (
        box_plot1_vaccine_html, 
        box_plot1_death_html,
        hist_vaccine_html, 
        hist_death_html, 
        line_vaccine_html, 
        line_death_html, 
        scatter_html, 
        rolling_7_html, 
        rolling_30_html, 
        rolling_180_html,
        fig_dif_html
    ) = creating_interactive_graphs1(data_clean, selected_country)

    #get content message
    content = about_eda()
    if content is None:
        return "<h1>Error: Failed to load about data content</h1>"

    # renders the html
    rendered_html = render_template_string("""
            <h1>Data Loaded and Explored</h1>
            <h2>Logging Steps (first 5):</h2>
            <pre>{{ log_content | join('\n') }}</pre>  <!-- Show the logs using Jinja filter -->

            <h2>First 5 Rows of Data Sets:</h2>
            <h3>Cleaned Data:</h3>
            <pre>{{ data_clean.head().to_html() | safe }}</pre>
            <h3>Outliers Removed by Standard Deviation:</h3>
            <pre>{{ data_nomean_out.head().to_html() | safe }}</pre>
            <h3>Outliers Removed by IQR:</h3>
            <pre>{{ data_nomedian_out.head().to_html() | safe }}</pre>
            <h3>Log-Transformed Data:</h3>
            <pre>{{ data_log.head().to_html() | safe }}</pre>

            <h2>Descriptive Statistics of Data Sets:</h2>
            <h3>Cleaned Data:</h3>
            <pre>{{ data_clean.describe().to_html() | safe }}</pre>
            <h3>Outliers Removed by Standard Deviation:</h3>
            <pre>{{ data_nomean_out.describe().to_html() | safe }}</pre>
            <h3>Outliers Removed by IQR:</h3>
            <pre>{{ data_nomedian_out.describe().to_html() | safe }}</pre>
            <h3>Log-Transformed Data:</h3>
            <pre>{{ data_log.describe().to_html() | safe }}</pre>

            <!-- Dropdown form for country selection -->
            <form method="POST">
                <label for="country">Select Country:</label>
                <select name="country" id="country">
                    <option value="">All Countries</option>
                    {% for country in unique_countries %}
                        <option value="{{ country }}" {% if country == selected_country %} selected {% endif %}>
                            {{ country }}
                        </option>
                    {% endfor %}
                </select>
                
                <label for="period">Select Period:</label>
                <select name="period" id="period">
                    <option value="7" {% if selected_period == '7' %} selected {% endif %}>Weekly</option>
                    <option value="30" {% if selected_period == '30' %} selected {% endif %}>Monthly</option>
                    <option value="180" {% if selected_period == '180' %} selected {% endif %}>Biannual</option>
                </select>
                
                <input type="submit" value="Filter">
            </form>

            <h1>Interactive Graphs</h1>
            <div>{{ box_plot1_vaccine_html | safe }}</div>
            <div>{{ box_plot1_death_html | safe }}</div>
            <div>{{ hist1_vaccine_html | safe }}</div>
            <div>{{ hist1_death_html | safe }}</div>
            <div>{{ line1_vaccine_html | safe }}</div>
            <div>{{ line1_death_html | safe }}</div>
            <div>{{ scatter1_html | safe }}</div>
            <div id="rolling_7_day">
                <h3>7-Day Rolling Mean</h3>
                {{ rolling_7_html | safe }}
            </div>
            <div id="rolling_30_day">
                <h3>30-Day Rolling Mean</h3>
                {{ rolling_30_html | safe }}
            </div>
            <div id="rolling_180_day">
                <h3>180-Day Rolling Mean</h3>
                {{ rolling_180_html | safe }}
            </div>
            <div>{{ fig_dif_html | safe }}</div>
                                           <!-- About Section -->
        <div id="about_eda">
            <h3>About the EDA</h3>
            {{ content | safe }}
        </div>

        """, 
        log_content=log_content,
        data_clean=data_clean,
        data_nomean_out=data_nomean_out,
        data_nomedian_out=data_nomedian_out,
        data_log=data_log,
        unique_countries=unique_countries,
        selected_country=request.form.get('country'),
        selected_period=request.form.get('period', '7'),  # Default to 7 for weekly
        box_plot1_vaccine_html=box_plot1_vaccine_html,
        box_plot1_death_html=box_plot1_death_html,
        hist1_vaccine_html=hist_vaccine_html,
        hist1_death_html=hist_death_html,
        line1_vaccine_html=line_vaccine_html,
        line1_death_html=line_death_html,
        scatter1_html=scatter_html,
        rolling_7_html=rolling_7_html,
        rolling_30_html=rolling_30_html,
        rolling_180_html = rolling_180_html,
        fig_dif_html = fig_dif_html,
        content = content)

    pdf = weasyprint.HTML(string=rendered_html).write_pdf()

        # Send the PDF as a downloadable file
    return send_file(
            BytesIO(pdf), 
            as_attachment=True, 
            download_name="cleaning_report.pdf", 
            mimetype='application/pdf'
        )

if __name__ == "__main__":
    app.run(debug=True)
