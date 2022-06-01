import pandas as pd
from arango import ArangoClient

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import minmax_scale
import networkx as nx

import plotly.graph_objects as go


import plotly.io as pio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import minmax_scale
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
from sklearn.preprocessing import minmax_scale
client = ArangoClient(hosts="http://localhost:8529")
sys_db = client.db('_system', username='root',password='root')
db = client.db("text", username="root", password="root")

pio.renderers.default='browser'

lemmas_from_sentences = pd.DataFrame(list(db.aql.execute('''for start_vertex in sentences
        for v, e in inbound start_vertex is_from
        filter e.type == 'lemmaToSent'
        collect sent = e._to, lemmas = v.lemma into groups ={
        "sentence" : e._to,
        "lemma" : v.lemma
        }
        return {"sentence":sent,
                "lemma":lemmas}
                '''))).groupby('sentence')['lemma'].apply(' '.join)

vectorizer = CountVectorizer(min_df=10)

termDocMatrix  = vectorizer.fit_transform(lemmas_from_sentences)

coOccurenceMatrix = termDocMatrix.T.dot(termDocMatrix)
# retire les liens d'un nodeà lui même dnas la matrice

coOccurenceMatrix.setdiag(0)

# construction du graphe
G = nx.from_scipy_sparse_array(coOccurenceMatrix,
                                parallel_edges=False)

# retire les arretes qui connectent un noeud à lui même 

G.remove_edges_from(nx.selfloop_edges(G))



# définition de la position des noeuds par spatialisation fruchterman reingold

FRL = nx.drawing.layout.fruchterman_reingold_layout(G,dim=3)

# kamada kawai
# KMK = nx.drawing.layout.kamada_kawai_layout(G)

for i in range(0,len(FRL)):
    G.nodes[i]['pos'] = FRL[i]
    


def make_edge_3d(x, y, z, width,scaledWidth):
    """
    Args:
        x: a tuple of the x from and to, in the form: tuple([x0, x1, None])
        y: a tuple of the y from and to, in the form: tuple([y0, y1, None])
        width: The width of the line

    Returns:
        a Scatter plot which represents a line between the two points given. 
    """
    return  go.Scatter3d(
                x=x,
                y=y,
                z=z,
                line=dict(width=width,color='#888'),
                hoverinfo='none',
                mode='lines',
                opacity=scaledWidth)




xTupleList = []
yTupleList = []
zTupleList = []

for ed in G.edges(): 
    xfrom = G.nodes()[ed[0]]['pos'][0]
    yfrom = G.nodes()[ed[0]]['pos'][1]
    zfrom = G.nodes()[ed[0]]['pos'][2]
    
    xto = G.nodes()[ed[1]]['pos'][0]
    yto = G.nodes()[ed[1]]['pos'][1]
    zto = G.nodes()[ed[1]]['pos'][2]

    
    xTupleList.append((xfrom,xto,None))
    yTupleList.append((yfrom,yto,None))
    zTupleList.append((zfrom,zto,None))
    

widthList = np.array([G.edges[ed]['weight'] for ed in G.edges()])

scaledWidthList = minmax_scale(widthList)



edge_trace = [make_edge_3d(x,y,z,w,sw) for x,y,z,w,sw in zip(xTupleList,
                                                          yTupleList,
                                                          zTupleList,
                                                          widthList*0.5,
                                                          scaledWidthList)]




node_x = []
node_y = []
node_z = []
for node in G.nodes():
    x, y, z = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)
    
    
    

node_trace = go.Scatter3d(
    x = node_x,
    y = node_y,
    z = node_z,
    text='text',
    textposition='top center',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title="Nombre d'occurences",
            xanchor='left',
            titleside='right'
        ),
        line_width=2))




node_trace.marker.color = list(vectorizer.vocabulary_.values())
node_trace.text = list(vectorizer.vocabulary_.keys())


layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', # transparent background
    plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
    xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
    yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
)


fig = go.Figure(layout = layout)

# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)# Add node trace
fig.add_trace(node_trace)# Remove legend
fig.update_layout(showlegend = False)