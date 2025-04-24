# app.py
from dash import Dash, html
import base64

# importing files
import Capstone_Milestone_3_Poisson
import Capstone_Milestone_3_Random_Forest
import Capstone_Milestone_3_DNN
import Capstone_Milestone_3_Feature_Engineering
pdf_doc = 'Capstone_Milestone_3_Documentation.pdf'
encoded_pdf = base64.b64encode(open(pdf_doc, 'rb').read()).decode()
pdf_doc_gray = 'Capstone_Milestone_3_Documentation_Grayscale.pdf'
encoded_pdf_gray = base64.b64encode(open(pdf_doc_gray, 'rb').read()).decode()

# Dash app layout
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Welcome to My Data Product"),
    html.P("This is deployed on Render!"),

    html.H2("Project Documentation (Color)"),
    html.Iframe(src=f"data:application/pdf;base64,{encoded_pdf}", style={"width": "100%", "height": "500px"}),

    html.H2("Project Documentation (Grayscale)"),
    html.Iframe(src=f"data:application/pdf;base64,{encoded_pdf_gray}", style={"width": "100%", "height": "500px"}),
])

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)