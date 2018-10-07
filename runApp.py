import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter, Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from models.HMsgClasses import getTokenizedMessage

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
v_data = pd.read_sql_table('tabMessagesTokenized', engine)

# load model
v_model = joblib.load("models/savedModel.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    v_group = v_data.drop(['id', 'message', 'original', 'messageTokenized'], axis = 1)
    v_shape = v_group.groupby('genre').agg({'genre': ['count']}).reset_index()
    v_shape.columns = ['genre', 'count']
    
    v_merge = pd.DataFrame()
    for item in v_group.drop('genre', axis = 1).columns:
        v_item  = v_group.groupby(['genre', item]).agg({item: ['count']}).reset_index()
        v_item.columns = ['genre', 'value', 'count']
        v_item['item'] = item
        v_item['value %'] = v_item[['genre', 'count']].apply(lambda x: round(x['count'] / v_shape[v_shape['genre'] == x['genre']]['count'].values[0] * 100, 2), axis = 1)    
        v_merge = pd.concat([v_merge, v_item])
    v_merge = v_merge[['genre', 'item', 'value', 'value %', 'count']].sort_values(['genre', 'value'])
    
    
    v_graphs = []
    v_graphs.append({ 'data': [ Bar( x = v_shape['genre'],
                                     y = v_shape['count'] ) ],
                      'layout': { 'title': 'Distribution of Message Genres (all categories)',
                                  'yaxis': { 'title': "Count" },
                                  'xaxis': { 'title': "Genre" } }})
    
    v_traces = []
    for item in v_merge['genre'].unique().tolist():
        for value in v_merge['value'].unique().tolist():
            v_traces.append( Scatter( x = v_merge[(v_merge['genre'] == item) & (v_merge['value'] == value)]['item'],
                                      y = v_merge[(v_merge['genre'] == item) & (v_merge['value'] == value)]['value %'],
                                      mode = 'lines+markers',
                                      name = f'Genre {item} - Value = {value}',
                                      marker = dict( size = 10,
                                                     line = dict(width = 1) ) ))
        v_graphs.append({ 'data': v_traces,
                          'layout': { 'title': f'Distribution of Message Genres <<{item}>>',
                                      'yaxis': { 'title': "Count" } }})
        v_traces = []
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(v_graphs)]
    graphJSON = json.dumps(v_graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids = ids, graphJSON = graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    v_query = request.args.get('query', '') 
    
    # use model to predict classification for query    
    v_labels = v_model.predict(getTokenizedMessage(v_query))[0]    
    v_classes = v_model.named_steps['classifier'].getClasses()
    v_results = dict(zip(v_classes, v_labels))
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query = v_query,
        classification_result = { 'query':   v_query, 
                                  'classif': v_results }
    )


def main():
    if ( len(sys.argv) == 2
         and sys.argv[1] == 'LOCALHOST' ):
        app.run(host = '127.0.0.1', port = 5000, debug = True)
    else:    
        app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()