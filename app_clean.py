from flask import Flask, render_template_string
from markupsafe import Markup
from Clean_Folder.Capstone_Milestone_3_Clean import load_data, log_step, log_content, about_data, scrub_data, outliers, scrub_new_data, about_cleaning, save_cleaned_data

# loads packages

app = Flask(__name__)

@app.route('/')
def clean():
    log_step("Starting data cleaning") 
    # Logs cleaning start
    
    data = load_data()
    # loads data
    if data is None:
        log_step("Failed to load data")
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to load data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content))) 

    content = about_data()
    if content is None:
        return "<h1>Error: Failed to load about data content</h1>"

    # Use Markup to ensure the content is rendered as HTML
    content_markup = Markup(content)

    # Debugging: Check the content being passed to the template
    print(f"Content to render: {content_markup}")

    scrubbed_data = scrub_data(data)
    if scrubbed_data is None:
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to scrub data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content)))
    save_cleaned_data(scrubbed_data, "Clean_Folder/data_clean.csv")

    # Process outliers
    data_nomean, data_nomedian, data_lg = outliers(scrubbed_data)
    if data_nomean is None or data_nomedian is None or data_lg is None:
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to process outliers. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content)))
    
    data_nomean_out = scrub_new_data(data_nomean)
    if data_nomean_out is None:
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to scrub data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content)))
    save_cleaned_data(data_nomean_out, "Clean_Folder/data_nomean_out.csv")

    data_nomedian_out = scrub_new_data(data_nomedian)
    if data_nomedian_out is None:
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to scrub data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content)))
    save_cleaned_data(data_nomedian_out, "Clean_Folder/data_nomedian_out.csv")

    data_log = scrub_new_data(data_lg)
    if data_log is None:
        return render_template_string("""
            <h1>Error</h1>
            <p>Failed to scrub data. Please check the logs for more details.</p>
            <h2>Log Content:</h2>
            <pre>{}</pre>
        """.format('\n'.join(log_content)))
    save_cleaned_data(data_log, "Clean_Folder/data_log.csv")
    log_step("Data Cleaned")

    content2 = about_cleaning()
    if content2 is None:
        return "<h1>Error: Failed to load about data content</h1>"

    # Use Markup to ensure the content is rendered as HTML
    content_markup_2 = Markup(content2)  # Changed the variable name to content_markup_2

    # Only render the success page if no errors
    rendered_html = render_template_string(f"""
        <h1>Data Loaded, Scrubbed, and Processed</h1>
        <h2>Logging Steps:</h2>
        <pre>{'\n'.join(log_content)}</pre>
        <h2>About the Data:</h2>  
        <p>{content_markup}</p>
        <h2>First 10 Rows of Scrubbed Data:</h2>
        <pre>{scrubbed_data.head()}</pre>
        <h2>Outliers Removed by Standard Deviation:</h2>
        <pre>{data_nomean_out.head()}</pre>
        <h2>Outliers Removed by IQR:</h2>
        <pre>{data_nomedian_out.head()}</pre>
        <h2>Log-Transformed Data (if outliers detected):</h2>
        <pre>{data_log.head()}</pre>
        <h2>About the Cleaning:</h2>  
        <p>{content_markup_2}</p>
    """)

    # Log the rendered HTML (for debugging)
    print(rendered_html)
    
    return rendered_html
     
if __name__ == "__main__":
    app.run(debug=True, port = 5000)

