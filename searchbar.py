 
    
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc

from app_functions import get_syntagmatic_data, plot_word_of_interest, concordancier, get_co_occurence_plot, extract_clickdata, placeholder_plot
from arango import ArangoClient


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.MINTY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4('Explorateur de texte',
                    className= 'text-center'),
            html.Div("Cliquer sur un point dans la vue d'ensemble pour observer les mots qui y sont associés et dans quels documents. \n Cliquer sur les mots reliés pour scroller les phrases où ces mots apparaissent ensembles.",style={'whiteSpace': 'pre-line'},
                     className = 'text-center')
        ])
    ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='co-occurence_graph',
              figure = get_co_occurence_plot())],
                width=7),
        dbc.Col([dcc.Graph(id='syntagmatic_graph')    
                ],width=5),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4(id='selection_status',style={'whiteSpace': 'pre-line'})
        ]),
        dbc.Col([
            html.H4('Recherche dans le texte',
                    id='info_search',
                    style={'whiteSpace': 'pre-line'})             
        ])
    ]),
    dbc.Row([
        dbc.Col([
                html.Div(id='double_input_text', style={'whiteSpace': 'pre-line'})
        ]),
        dbc.Col([
               dcc.Input(id="recherche", type="text", placeholder="Recherche", debounce=True,
                        style={'width':"100%"}),
               dcc.Dropdown(['NYC', 'MTL', 'SF'], 'NYC', id='demo-dropdown',multi=True)
        ])
    ])
], fluid=True)



@app.callback(
    Output('syntagmatic_graph','figure'),
    Input('co-occurence_graph','clickData')
)

## todo :

### word of interest renvoie des tokens identiques au lemme mais pas les tokens contractés sous ce lemme
def plot_syntagmatic(value):
    if value is None:
        return placeholder_plot("Mots reliés","Sélectionner un point sur la vue d'ensemble pour explorer ses relations.")
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

@app.callback(
    Output('selection_status','children'),
[
    Input('co-occurence_graph', 'clickData'),
    Input('syntagmatic_graph','clickData')
    ])

def update_status(input1,input2):
     if input1 is not None and input2 is not None :
        
        return (f'Mot racine : {extract_clickdata(input1)} | Mot relié : {extract_clickdata(input2)}')
     else :
        return ('Relation non sélectionnée.')

if __name__ == '__main__':
    app.run_server(debug=False)
