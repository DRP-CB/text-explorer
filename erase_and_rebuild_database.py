import spacy
import os
from os import path
import time
import glob
import pandas as pd
from arango import ArangoClient

# connexion à la base de données
client = ArangoClient(hosts="http://localhost:8529")
sys_db = client.db('_system', username='root',password='root')

sys_db.delete_database('text')


sys_db.create_database('text')
db = client.db("text", username="root", password="root")
graph = db.create_graph('text_explorer')
# création des collections
docs = graph.create_vertex_collection('docs')
sentences = graph.create_vertex_collection('sentences')
tokens = graph.create_vertex_collection('tokens')
lemmas = graph.create_vertex_collection('lemmas')
# création des arrêtes
is_from = graph.create_edge_definition(
    edge_collection='is_from',
    from_vertex_collections=['sentences','tokens'],
    to_vertex_collections=['docs','sentences']
)
contracts_to = graph.create_edge_definition(
    edge_collection='contracts_to',
    from_vertex_collections=['tokens'],
    to_vertex_collections=['lemmas']
)
syntagmatic_link = graph.create_edge_definition(
    edge_collection='syntagmatic_link',
    from_vertex_collections=['tokens'],
    to_vertex_collections=['tokens']
)
