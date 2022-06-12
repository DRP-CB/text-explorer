 
    
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc

from app_functions import get_syntagmatic_data, plot_word_of_interest, concordancier, get_co_occurence_plot, extract_clickdata, placeholder_plot, search_tokens,  df_to_text
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
               dcc.Input(id="recherche", type="text", placeholder="Rechercher un ou plusieurs mots séparés par un espace.",
                        style={'width':"100%"},
                        debounce =True),
               html.Div(id='research-results',style={'whiteSpace': 'pre-line'})
        ])
    ])
], fluid=True)




@app.callback(
    Output('syntagmatic_graph','figure'),
    Input('co-occurence_graph','clickData')
)

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

@app.callback(
    Output('research-results','children'),
    [Input('recherche','value')])

def search(input1):
    
    if input1 is None :
        return 'En attente de termes de recherche.'
    elif input1 is not None :
        df_search = search_tokens(input1.split(' '))
        if df_search is None :
            return "Mot non trouvé dans le corpus."
        elif df_search.shape[0] == 0:
            return "Un des mots recherchés est absent du corpus."
        elif df_search.shape[0] > 0:
            return df_to_text(df_search)
    
if __name__ == '__main__':
    app.run_server(debug=True)
