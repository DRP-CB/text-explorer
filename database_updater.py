# à refactoriser, un point de départ pour travailler l'UI

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

if not sys_db.has_database('text'):
    
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
else:
    db = client.db("text", username="root", password="root")
    

def get_text(path):
    with open(path, encoding='utf8') as f:
        return(f.read().replace('\n',' '))
        f.close()
        
def get_filename_from_path(path):
    return os.path.normpath(path).split(os.sep)[-1]

files = glob.glob('/home/paul/projects/text_for_app/*.{}'.format('txt'))


documents = pd.DataFrame({'filepath':files,
                          'doc_name':[get_filename_from_path(filepath) for filepath in files],
                          'doc_number':list(range(0,len(files)))})

def insert_docs(documents_found):
    in_db_doclist = pd.DataFrame(list(db.aql.execute('''FOR doc in docs RETURN doc''')))
    if in_db_doclist.shape == (0,0):
        pass
    else :
        documents_found = documents_found[~documents_found['doc_name'].isin(in_db_doclist['doc_name'])]
        last_doc_number = in_db_doclist['doc_number'].max()
        documents_found['doc_number'] = documents_found['doc_number'] + last_doc_number + 1
    
    dict_list_documents_to_insert = []
    for doc_name, number, path  in zip(documents_found['doc_name'], documents_found['doc_number'], documents_found['filepath']):
        dict_list_documents_to_insert.append({'_key':f'doc{number}',
                                                'doc_name':doc_name,
                                                'doc_path':path,
                                                'doc_number':number,
                                                'processed':'False'})
    db.collection('docs').import_bulk(dict_list_documents_to_insert)
    
insert_docs(documents)

nlp = spacy.load('fr_dep_news_trf')

texts_to_process_df = pd.DataFrame(
                        list(
                            db.aql.execute('''
                                            FOR doc in docs
                                            FILTER doc.processed == 'False'
                                            RETURN {path :doc.doc_path, number : doc.doc_number}
                                            ''')
                            )
                        )

processed_docs = list(nlp.pipe([get_text(path) for path in texts_to_process_df['path']]))


def insert_sentences(processed_doc, doc_number):
    dict_sentences_to_insert = []
    for sentence_number, sentence in enumerate(processed_doc.sents):
        if sentence.text != ' ':
            dict_sentences_to_insert.append({'_key':f'doc{doc_number}sent{sentence_number}',
                                             'content':sentence.text})
        else:
            pass
    db.collection('sentences').import_bulk(dict_sentences_to_insert)
    
for processed_text, doc_number in zip(processed_docs,texts_to_process_df['number']):
    insert_sentences(processed_text,doc_number)
    
def get_vocab_table_from_text(processed_text):
    tokens, lemmas = [], []
    for token in processed_text:
        if not token.is_punct and not token.is_stop and not token.is_space and not token.is_digit:
            tokens.append(token.text.lower())
            lemmas.append(token.lemma_.lower())
    vocab_table = pd.DataFrame({'token':tokens,
                                'lemma':lemmas})
    return vocab_table

def unique_vocabulary(token_lemma_table,word_type):
    return(token_lemma_table[word_type].drop_duplicates().reset_index(drop=True))


def get_vocab_in_db():
    vocab_tokens_from_db = pd.DataFrame(
        list(db.aql.execute("""
            FOR doc in tokens
            RETURN {word :doc.token, key : doc._key}
            """))
        )
        
    vocab_lemmas_from_db = pd.DataFrame(
        list(db.aql.execute("""
            FOR doc in lemmas
            RETURN {word :doc.lemma, key : doc._key}
            """))
        )
    return vocab_tokens_from_db, vocab_lemmas_from_db

def keep_only_new_vocab(vocab_from_text,vocab_from_db):
    new_vocab = vocab_from_text[~vocab_from_text.isin(vocab_from_db['word'])].reset_index(drop=True)
    return new_vocab

def add_vocab_to_db(processed_text):
    token_to_lemma_table = get_vocab_table_from_text(processed_text)
    
    text_tokens_series = unique_vocabulary(token_to_lemma_table,'token')
    text_lemmas_series = unique_vocabulary(token_to_lemma_table,'lemma')
    
    tokens_from_db, lemmas_from_db = get_vocab_in_db()
    
    if tokens_from_db.shape == (0,0):
        dict_list_tokens_to_insert = []
        for index, token in zip(text_tokens_series.index,text_tokens_series.values):
            dict_list_tokens_to_insert.append({"_key":f'token{index}',
                                               'token':token})
        db.collection('tokens').import_bulk(dict_list_tokens_to_insert)

        dict_list_lemmas_to_insert = []
        for index, lemma in zip(text_lemmas_series.index,text_lemmas_series.values):
            dict_list_lemmas_to_insert.append({"_key":f'lemma{index}',
                                               'lemma':lemma})
        db.collection('lemmas').import_bulk(dict_list_lemmas_to_insert)

    else :
        new_vocab_tokens = keep_only_new_vocab(text_tokens_series,tokens_from_db)
        new_vocab_lemmas = keep_only_new_vocab(text_lemmas_series,lemmas_from_db)

        new_vocab_tokens.index = new_vocab_tokens.index + tokens_from_db.shape[0] + 1
        new_vocab_lemmas.index = new_vocab_lemmas.index + lemmas_from_db.shape[0] + 1

        dict_list_tokens_to_insert = []
        for index, token in zip(new_vocab_tokens.index,new_vocab_tokens.values):
            dict_list_tokens_to_insert.append({"_key":f'token{index}',
                                               'token':token})
        db.collection('tokens').import_bulk(dict_list_tokens_to_insert)

        dict_list_lemmas_to_insert = []
        for index, lemma in zip(new_vocab_lemmas.index,new_vocab_lemmas.values):
            dict_list_lemmas_to_insert.append({"_key":f'lemma{index}',
                                               'lemma':lemma})
        db.collection('lemmas').import_bulk(dict_list_lemmas_to_insert)
        
for doc in processed_docs:
    add_vocab_to_db(doc)


sentences_keys = pd.Series(list(db.aql.execute('''
                        FOR doc in sentences
                        return doc._key
                        ''')))

doc_number_for_sentences = sentences_keys.str.extract('(\d+)')[0]
sentences_number = sentences_keys.str.extract('\D+\d+\D+(\d+)')[0]

dict_is_from_sent_doc_to_insert = []
for sentence_key, doc_number, sentence_number in zip(sentences_keys.values,doc_number_for_sentences,sentences_number.values):
    dict_is_from_sent_doc_to_insert.append({'_from':sentence_key,
                                            '_to':f'doc{doc_number}',
                                            'sentence_number': sentence_number})
db.collection('is_from').import_bulk(dict_is_from_sent_doc_to_insert,
                                     from_prefix='sentences/',
                                     to_prefix='docs/')

def create_dependancy_df(processed_text,doc_number):
    token_text, token_dep, token_head_text, token_head_pos, sentence_number = [], [], [], [],[]

    for count, sentence in enumerate(processed_text.sents):
        for token in sentence:
            if not token.is_punct and not token.is_stop and not token.is_space and not token.is_digit:
                token_text.append(token.text)
                token_dep.append(token.dep_), 
                token_head_text.append(token.head.text), 
                token_head_pos.append( token.head.pos_)
                sentence_number.append(f'doc{doc_number}sent{count}')

    
    df = pd.DataFrame({'token':token_text,
                       'dep':token_dep,
                       'head_text':token_head_text,
                       'head_pos':token_head_pos,
                       'sentence_number':sentence_number})   
    df = df[df['head_pos']!="SPACE"]
    
    return df


def create_dependancy_df(processed_text,doc_number):
    token_text, token_lemma, token_dep, token_head_text, token_head_pos, sentence_number = [], [], [], [], [], []

    for count, sentence in enumerate(processed_text.sents):
        for token in sentence:
            if not token.is_punct and not token.is_stop and not token.is_space and not token.is_digit:
                token_text.append(token.text)
                token_lemma.append(token.lemma_)
                token_dep.append(token.dep_), 
                token_head_text.append(token.head.text), 
                token_head_pos.append( token.head.pos_)
                sentence_number.append(f'doc{doc_number}sent{count}')

    
    df = pd.DataFrame({'token':token_text,
                       'lemma':token_lemma,
                       'dep':token_dep,
                       'head_text':token_head_text,
                       'head_pos':token_head_pos,
                       'sentence_number':sentence_number})   
    df = df[df['head_pos']!="SPACE"]
    
    return df


dependancy_dfs = []
for count, df in enumerate(processed_docs):
    dependancy_dfs.append(create_dependancy_df(df,count))
    
tokens_db, lemmas_db = get_vocab_in_db()


def insert_dependancies(dependancy_df, vocab_table_tokens_in_db,lemmas_in_db):
    token_from_sentence_table = dependancy_df.merge(vocab_table_tokens_in_db, 
                                                        left_on='token', 
                                                        right_on='word', sort = 'outer')\
                                                        .rename(columns={'key':'token_key'})\
                                                        .drop('word',axis=1)
    lemma_from_sentence_table = dependancy_df.merge(lemmas_in_db, 
                                                        left_on='lemma', 
                                                        right_on='word', sort = 'outer')\
                                                        .rename(columns={'key':'lemma_key'})\
                                                        .drop('word',axis=1)
    
    token_from_sentence_table = dependancy_df.merge(vocab_table_tokens_in_db, 
                                                        left_on='token', 
                                                        right_on='word')\
                                                        .rename(columns={'key':'token_key'})\
                                                        .drop('word',axis=1)

    dependancy_table_for_insert = token_from_sentence_table.merge(vocab_table_tokens_in_db,
                                                        left_on='head_text',
                                                        right_on='word')\
                                                        .rename(columns={'key':'head_text_key'})\
                                                        .drop('word',axis=1)
    dict_is_from_sent_token_to_insert = []
    for token_key,sentence_number in zip(token_from_sentence_table['token_key'],token_from_sentence_table['sentence_number'] ):
        dict_is_from_sent_token_to_insert.append({'_from':token_key,
                                                '_to':sentence_number,
                                                'type':'tokenToSent'})
    db.collection('is_from').import_bulk(dict_is_from_sent_token_to_insert,
                                         from_prefix='tokens/',
                                         to_prefix='sentences/')
    
    dict_is_from_sent_lemma_to_insert = []
    for lemma_key,sentence_number in zip(lemma_from_sentence_table['lemma_key'],token_from_sentence_table['sentence_number'] ):
        dict_is_from_sent_lemma_to_insert.append({'_from':lemma_key,
                                                '_to':sentence_number,
                                                'type':'lemmaToSent'})
    db.collection('is_from').import_bulk(dict_is_from_sent_lemma_to_insert,
                                         from_prefix='lemmas/',
                                         to_prefix='sentences/')
    
    dict_syntagmatic_link_to_insert = []
    for head_text_key, token_key, dep_relation, head_pos_tag, sentence_number in zip(dependancy_table_for_insert['head_text_key'],
                                                                                     dependancy_table_for_insert['token_key'],
                                                                                     dependancy_table_for_insert['dep'],
                                                                                     dependancy_table_for_insert['head_pos'],
                                                                                     dependancy_table_for_insert['sentence_number']):
        dict_syntagmatic_link_to_insert.append({'_from':head_text_key,
                                                '_to':token_key,
                                                'dep_relation':dep_relation,
                                                'head_pos_tag':head_pos_tag,
                                                'from_sentence_number':sentence_number})
    db.collection('syntagmatic_link').import_bulk(dict_syntagmatic_link_to_insert,
                                                 from_prefix='tokens/',
                                                 to_prefix='tokens/')
    contracts_to_table = dependancy_table_for_insert.merge(lemmas_in_db, left_on='lemma', right_on='word')\
                        .rename(columns={'key':'lemma_key'})\
                        .drop('word',axis=1)
    dict_contracts_to = []
    for token_key, lemma_key, sentence_number in zip(contracts_to_table['token_key'],
                                                     contracts_to_table['lemma_key'],
                                                     contracts_to_table['sentence_number']):

        dict_contracts_to.append({'_from':token_key,
                                 '_to':lemma_key,
                                 'sentence_number':sentence_number})

    db.collection('contracts_to').import_bulk(dict_contracts_to,
                                                 from_prefix='tokens/',
                                                 to_prefix='lemmas/')

for df in dependancy_dfs:
    insert_dependancies(df, tokens_db, lemmas_db)