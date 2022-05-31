from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from text_explorer_functions import get_syntagmatic_data, plot_word_of_interest, concordancier
from arango import ArangoClient

client = ArangoClient(hosts="http://localhost:8529")
sys_db = client.db('_system', username='root',password='root')
db = client.db("text", username="root", password="root")

syntagmatic_plot = plot_word_of_interest('classe')


app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='syntagmatic_graph',
              figure = syntagmatic_plot),
    html.Div(id='graph_output', style={'whiteSpace': 'pre-line'}),

    dcc.Textarea(
        id='textarea-example',
        value='classe',
        style={'width': '100%', 'height': 300},
    ),
    html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'}),
    html.Div(id='double_input_text', style={'whiteSpace': 'pre-line'})
])


@app.callback(
    Output('graph_output','children'),
    Input('syntagmatic_graph','clickData')
)
def return_click(value):
    if value is None:
        return 'sélectionner un point'
    else :
        return value['points'][0]['text']
    

@app.callback(
    Output('double_input_text', 'children'),
    [
    Input('textarea-example', 'value'),
    Input('graph_output','children')
    ])

def return_phrase(input1, input2):
    concordancier_data = concordancier(input1,input2)
    return concordancier_data
    
    
    
@app.callback(
    Output('textarea-example-output', 'children'),
    Input('textarea-example', 'value')
)
def return_phrase(key):
    sent = list(db.aql.execute(f'''
    FOR sent in sentences
    FILTER sent._key == {"'"+ key +"'"}
    RETURN sent.content
'''))[0]
    return sent

if __name__ == '__main__':
    app.run_server(debug=True)
