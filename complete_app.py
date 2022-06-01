from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from text_explorer_functions import get_syntagmatic_data, plot_word_of_interest, concordancier, get_co_occurence_plot, extract_clickdata, placeholder_plot
from arango import ArangoClient

client = ArangoClient(hosts="http://localhost:8529")
sys_db = client.db('_system', username='root',password='root')
db = client.db("text", username="root", password="root")


app = Dash(__name__)

app.layout = html.Div([
    
    dcc.Graph(id='co-occurence_graph',
              figure = get_co_occurence_plot()),
    
    dcc.Graph(id='syntagmatic_graph'),
        
    html.Div(id='double_input_text', style={'whiteSpace': 'pre-line'})
])




@app.callback(
    Output('syntagmatic_graph','figure'),
    Input('co-occurence_graph','clickData')
)
def plot_syntagmatic(value):
    if value is None:
        return placeholder_plot("Sélectionner un point sur la vue d'ensemble pour explorer ses relations.")
    else :
        fig = plot_word_of_interest(extract_clickdata(value))
        return fig


@app.callback(
    Output('double_input_text', 'children'),
    [
    Input('co-occurence_graph', 'clickData'),
    Input('syntagmatic_graph','clickData')
    ])

def return_phrase(input1, input2):
    if input1 is not None and input2 is not None :
        
        concordancier_data = concordancier(extract_clickdata(input1),
                                           extract_clickdata(input2))
        return concordancier_data
    else :
        return ('Sélectionner un point dans les relations pour lire son contexte.')

    

if __name__ == '__main__':
    app.run_server(debug=True)
