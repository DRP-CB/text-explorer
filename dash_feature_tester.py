from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import glob 


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Text Explorer"),
    html.Button('Lancer Scan', id='scan-button'),
    html.Div(id='output-button',children='children')])
   


@app.callback(
    Output(component_id='output-button', component_property='children'),
    Input(component_id='scan-button', component_property='n_clicks')
)
def addNew(click):
    if click is not None :
        print('button clicked')
        files = glob.glob('/home/paul/projects/text_for_app/*.{}'.format('txt'))
        return files
        
    else:
        pass


if __name__ == '__main__':
    app.run_server(debug=True)

