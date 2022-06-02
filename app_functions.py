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

layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', # transparent background
    plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
    xaxis =  {'showgrid': False, 'zeroline': True}, # no gridlines
    yaxis = {'showgrid': False, 'zeroline': True}, # no gridlines
    )

def placeholder_plot(title,content):
         
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(
        x = [0],
        y = [0],
        mode = 'text',
        text=content))
    fig.update_layout(title=title)
    return fig

def extract_clickdata(data):
    return data['points'][0]['text']


def make_edge(x, y, width,scaledWidth):
    """
    Args:
        x: a tuple of the x from and to, in the form: tuple([x0, x1, None])
        y: a tuple of the y from and to, in the form: tuple([y0, y1, None])
        width: The width of the line
    Returns:
        a Scatter plot which represents a line between the two points given. 
    """
    return  go.Scatter(
                x=x,
                y=y,
                line=dict(width=width,color='#888'),
                hoverinfo='none',
                mode='lines',
                opacity=scaledWidth)


def get_co_occurence_plot():
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

    vectorizer = CountVectorizer(min_df=0.01)

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

    FRL = nx.drawing.layout.fruchterman_reingold_layout(G,
                                                       k=10/len(G))

    # kamada kawai
    #KMK = nx.drawing.layout.kamada_kawai_layout(G)

    for i in range(0,len(FRL)):
        G.nodes[i]['pos'] = FRL[i]




    xTupleList = []
    yTupleList = []

    for ed in G.edges(): 
        xfrom = G.nodes()[ed[0]]['pos'][0]
        yfrom = G.nodes()[ed[0]]['pos'][1]

        xto = G.nodes()[ed[1]]['pos'][0]
        yto = G.nodes()[ed[1]]['pos'][1]
        xTupleList.append((xfrom,xto,None))
        yTupleList.append((yfrom,yto,None))

    widthList = np.array([G.edges[ed]['weight'] for ed in G.edges()])

    scaledWidthList = minmax_scale(widthList)/1.5



    edge_trace = [make_edge(x,y,w,sw) for x,y,w,sw in zip(xTupleList,yTupleList,widthList*0.5,scaledWidthList)]




    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)




    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text='text',
        textposition='top center',
        marker=dict(
            showscale=False,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='reds',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Nombre d'occurences",
                xanchor='left',
                titleside='right'
            ),
            line_width=2))



    node_opacity = pd.Series(minmax_scale(list(vectorizer.vocabulary_.values())))
    node_opacity = node_opacity + 0.2
    node_opacity.clip(upper=1,inplace=True)
    node_trace.marker.size = minmax_scale(list(vectorizer.vocabulary_.values()))*15
    node_trace.marker.color =  minmax_scale(list(vectorizer.vocabulary_.values()))
    node_trace.text = list(vectorizer.vocabulary_.keys())
    node_trace.marker.opacity = node_opacity


    fig = go.Figure(layout = layout)

    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)# Add node trace
    fig.add_trace(node_trace)# Remove legend
    fig.update_layout(showlegend = False,
                      height=800,
                      title="Vue d'ensemble")
    
    return fig


def get_syntagmatic_data(word):
    
    doc_titles = pd.DataFrame(list(db.aql.execute(f"""for doc in docs
filter doc.doc_number in [1,2]
return {{'doc_title' : doc.doc_name,
        'doc_number' : doc.doc_number}}""")))
    
    data = pd.DataFrame(list(db.aql.execute(f'''
    
 for lemma in lemmas 
filter lemma.lemma == '{word}'
let selected_lemma = lemma._id

for v_lemma, e_lemma in 1..1
inbound selected_lemma
contracts_to
let selected_tokens = v_lemma

    for v, e in 1..1 
    any selected_tokens
    syntagmatic_link
    filter e.dep_relation != 'ROOT'

    return {{"token": v.token,
            "relation" : e.dep_relation,
            "from_sentence" : e.from_sentence_number,
            "head_pos_tag" : e.head_pos_tag
            }}

                    ''')))
    
     
   
    
    # à remplacer avec une requette donnant le titre du document
    if data.shape == (0,0):
        return "Mot non trouvé"
    else :
        data['doc'] = data['from_sentence'].str.extract('doc(\d+)')
        data['doc'] = data['doc'].astype(int)
        
        doclist = data['doc'].drop_duplicates().tolist()
        
        doc_titles = pd.DataFrame(list(db.aql.execute(f"""for doc in docs
filter doc.doc_number in {doclist}
return {{'doc_title' : doc.doc_name,
        'doc_number' : doc.doc_number}}""")))
        
        
        data = data.merge(doc_titles, left_on='doc', right_on='doc_number')
        #data['doc_title'] = data['doc_title'].str.extract('(\w+).')
        data.drop('doc',axis=1, inplace=True)
        data.rename(columns={'doc_title':'doc'},inplace=True)
        return data
    
    


def plot_word_of_interest(word_of_interest):
    
# récupération table de relations syntagmatiques depuis la db
    syntagmatic_data = get_syntagmatic_data(word_of_interest)
    
    if type(syntagmatic_data) == str :
        return placeholder_plot(syntagmatic_data)
    else:
        
        size_of_interest = 10

        syntagmatic_data = syntagmatic_data.sort_values('from_sentence')
    # ajout d'un 1 pour calculer les fréquences par aggrégation
        syntagmatic_data['count'] = 1

        # construction d'une edgelist pour créer le graphe
        edgelist = syntagmatic_data.groupby(['token','doc']).aggregate('sum')
        # l'index est multiple, on l'applatit
        edgelist = pd.DataFrame(edgelist.to_records())

        edgelist.rename(columns={"token":'from',
                                      'doc':'to'}, inplace=True)
        # on calcule la fréquence sur le lien racine / document
        edgelist_token_doc = pd.DataFrame(edgelist.groupby('to').aggregate('sum').to_records())
        edgelist_token_doc['from'] = word_of_interest

        # concaténation des docs et mots en une edgelist
        full_edgelist = pd.concat([edgelist, edgelist_token_doc])

        # création du graphe et calcul de la spatialisation
        G = nx.from_pandas_edgelist(full_edgelist,'from','to',edge_attr='count')
        KKL = nx.drawing.layout.kamada_kawai_layout(G)

        # insertion de la spatialisation pour la construction des arrêtes
        x_from_to = []
        y_from_to = []
        for from_vertex, to_vertex in zip(full_edgelist['from'], full_edgelist['to']):
            x_from_to.append((KKL[from_vertex][0], KKL[to_vertex][0]))
            y_from_to.append((KKL[from_vertex][1], KKL[to_vertex][1]))

        full_edgelist['from_and_to_x'] = x_from_to
        full_edgelist['from_and_to_y'] = y_from_to
        full_edgelist.reset_index(inplace=True,drop=True)

        # construction d'une échelle allant de 0.1 à 1 pour la transparence et épaisseur des arrêtes
        mask = full_edgelist['from'] == word_of_interest
        indexes_root_word, indexes_doc_words = full_edgelist[mask].index, full_edgelist[~mask].index

        full_edgelist.loc[indexes_root_word,'adjusted_count'] = minmax_scale(full_edgelist['count'][indexes_root_word]) +1
        full_edgelist.loc[indexes_doc_words,'adjusted_count'] = minmax_scale(full_edgelist['count'][indexes_doc_words]) +1

        # Sans le +0.1 le minmax donne 0 et donc pas de ligne. Si on ajout 0.1 on sort du range 0 - 1 autorisé par la valeur de transparence

        full_edgelist['opacity'] = minmax_scale(full_edgelist['adjusted_count'])+0.1
        full_edgelist['opacity'] = full_edgelist['opacity'].clip(upper=1)

        # création de la disposition des arrêtes
        edge_trace = [make_edge(x,y,w,sw) for x,y,w,sw in zip(full_edgelist['from_and_to_x'],
                                                              full_edgelist['from_and_to_y'],
                                                              full_edgelist['adjusted_count'],
                                                              full_edgelist['opacity'])]

        # Disposition des noeuds
        df_KKL = pd.DataFrame(KKL).T
        df_KKL.columns = ['x','y']

        # coloration en fonction du mot racine, document et mot relié

        df_KKL['color'] = 'rgb(255,102,102)'

        df_KKL.loc[word_of_interest,'color'] = 'rgb(0,204,0)'


        # A MODIFIER : besoin d'un filtre capable de repérer les documents finit par .txt ? 
        mask_doc = df_KKL.index.str.contains('.txt')




        df_KKL.loc[df_KKL.index[mask_doc],'color'] = 'rgb(0,128,255)'
        
        ######
        # faire un merge de ces infos sur df_KKL et ajouter le word of interest avec une tailel arbitraire
        ######
        doc_count = syntagmatic_data.groupby('doc').sum('count')
        doc_count['count'] =( minmax_scale(doc_count['count']) +1) *5
        token_count = syntagmatic_data.groupby('token').sum('count')
        token_count['count'] = (minmax_scale(token_count['count']) +1) *5

        df_count = pd.concat([doc_count,token_count,pd.DataFrame({'count':[size_of_interest]},index=[word_of_interest])])
        
        df_KKL = df_KKL.join(df_count)
        node_trace = go.Scatter(
        x=df_KKL['x'], y=df_KKL['y'],
        mode='markers+text',
        textposition='top center',
        marker_color=df_KKL['color'],
        marker_size=df_KKL['count'])
        node_trace.text = df_KKL.index.str.replace('.txt','',regex=False)
        
        fig = go.Figure(layout=layout)
        for trace in edge_trace:
            fig.add_trace(trace)# Add node trace
        fig.add_trace(node_trace)
        # Remove legend
        fig.update_layout(showlegend = False,
                          height=800,title='Mots reliés')
        return fig
    
    
        
    
    
    
    
    
def concordancier(root_word, dep_word):
    df = get_syntagmatic_data(root_word)
    df_sentences = df[df['token'] == dep_word][['from_sentence','doc']]
    sentences = [sent['content'] for sent in db.collection('sentences').get_many(df_sentences['from_sentence'])]
    df_sentences['sentences'] = sentences
    df_sentences.drop_duplicates(inplace=True)
    list_sentences = []
    for sentence, doc, content in zip(df_sentences['from_sentence'],
                                      df_sentences['doc'],
                                      df_sentences['sentences']):
        list_sentences.append(f'Tiré du document : {doc} | ID de la phrase : {sentence} \n __ \n \n {content} \n __ \n \n')
    return ' '.join(list_sentences)