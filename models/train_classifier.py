import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine, MetaData, Table

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib

from .HMsgClasses import messageTokenize
from .HMsgClasses import HMsgExtractMessage
from .HMsgClasses import HMsgCountVectorizer
from .HMsgClasses import HMsgTfidfTransformer
from .HMsgClasses import HMsgFeatureExtract
from .HMsgClasses import HMsgClassifier

def load_data(p_database_filepath, p_reloadData = False):
    v_engine = create_engine(f'sqlite:///{p_database_filepath}')
    v_metadata = MetaData(v_engine, reflect = True) 
    
    v_tabName = 'tabMessagesTokenized'
    if ( p_reloadData 
         or v_tabName not in v_metadata.tables.keys() ):
        if v_tabName in v_metadata.tables.keys():
            v_table = Table(v_tabName, v_metadata)
            v_table.drop(v_engine)

        v_data = pd.read_sql_table('tabMessages', v_engine)

        v_count = 0
        for idx in v_data.index:
            v_token = messageTokenize(v_data.loc[idx, 'message'])
            v_data.loc[idx, 'messageTokenized'] = v_token[0]
            v_data.loc[idx, 'flag_first_verb']  = v_token[1]
            v_data.loc[idx, 'flag_last_verb']   = v_token[2]
            v_data.loc[idx, 'flag_first_nnp']   = v_token[3]
            v_data.loc[idx, 'flag_last_nnp']    = v_token[4]
            v_count += 1
            if v_count % 600 == 0: print(f'Rows processed: {v_count} / {v_data.shape[0]}')

        v_data.to_sql(v_tabName, v_engine, index = False)
    else:
        v_data = pd.read_sql_table(v_tabName, v_engine) 
        
    print(f'    Database loaded.')  
    
    v_cols = ['messageTokenized', 'flag_first_verb', 'flag_last_verb', 'flag_first_nnp', 'flag_last_nnp']

    v_mapGenre = { 'news':    0,
                   'direct':  1,
                   'social':  2 }
    v_target = v_data.drop(v_cols, axis = 1).drop(['id', 'message', 'original'], axis = 1)
    v_target['genre'] = v_target['genre'].map(v_mapGenre)
    
    # We check that all the classes have at least one value specified
    v_sum = v_target.sum(axis = 0)
    v_target.drop(v_sum[v_sum == 0].index, axis = 1, inplace = True)
    
    return v_data[v_cols], v_target, v_mapGenre


def build_model(p_CVSplits, p_pointsBin, p_maxCateg, p_maxFeatures, p_debug):
    return Pipeline([ ('features', FeatureUnion([ ('text_pipeline', Pipeline([ ('step_01', HMsgExtractMessage()),
                                                                               ('step_02', HMsgCountVectorizer( max_features = p_maxFeatures )),
                                                                               ('step_03', HMsgTfidfTransformer()) ]) ),
                                                  ('feat_01', HMsgFeatureExtract('flag_first_verb')),
                                                  ('feat_02', HMsgFeatureExtract('flag_last_verb')),
                                                  ('feat_03', HMsgFeatureExtract('flag_first_nnp')),
                                                  ('feat_04', HMsgFeatureExtract('flag_last_nnp')),
                                               ])),
                      ('classifier', HMsgClassifier(p_CVSplits, p_pointsBin, p_maxCateg, p_debug)) ])

def evaluate_model(p_model, p_X, p_y, p_classes):
    y_pred = p_model.predict(p_X)
    p_model.named_steps[p_model.steps[-1][0]].classificationReport(p_y, y_pred, p_classes)
    return y_pred

def save_model(p_model, p_model_filepath):
    joblib.dump(p_model, p_model_filepath)
    joblib.dump(p_model.named_steps['features'],   f'{p_model_filepath.split(".pkl")[0]}_features.pkl')
    joblib.dump(p_model.named_steps['classifier'], f'{p_model_filepath.split(".pkl")[0]}_classifier.pkl')
    
    v_vectorizer = p_model.named_steps['features'].transformer_list[0][1].named_steps['step_02']
    joblib.dump(v_vectorizer, f'{p_model_filepath.split(".pkl")[0]}_vectorizer.pkl')
    
    v_transformer = p_model.named_steps['features'].transformer_list[0][1].named_steps['step_03']
    joblib.dump(v_transformer, f'{p_model_filepath.split(".pkl")[0]}_transformer.pkl')
    
    return

def executeMain( p_database_filepath, 
                 p_model_filepath, 
                 p_CVSplits     = 12, 
                 p_pointsBin    = 15, 
                 p_maxCateg     = None, 
                 p_maxFeatures  = 12000, 
                 p_reloadData   = False, 
                 p_debug        = False ):
    print('Loading data...')
    print(f'    Database filepath <<{p_database_filepath}>>')        
    v_X, v_y, v_mapGenre = load_data(p_database_filepath, p_reloadData)
    X_train, X_test, y_train, y_test = train_test_split(v_X, v_y, test_size = 0.10, random_state = 42)
    
    print('Building model...')
    v_model =  build_model(p_CVSplits, p_pointsBin, p_maxCateg, p_maxFeatures, p_debug)
        
    print('Training model...')
    v_model.fit(X_train, y_train)
        
    print('Evaluating model...')
    y_pred_test = evaluate_model(v_model, X_test, y_test, None)

    print('Saving model...')
    print(f'    Model filepath <<{p_model_filepath}>>')
    save_model(v_model, p_model_filepath)

    print('Trained model saved!')
    return v_model, X_train, X_test, y_train, y_test, y_pred_test 

def main():
    if len(sys.argv) == 3:
        v_database_filepath, v_model_filepath = sys.argv[1:]
        _ = executeMain( p_database_filepath = v_database_filepath, 
                         p_model_filepath    = v_model_filepath,
                         p_CVSplits = 3,
                         p_pointsBin = 15,
                         p_maxFeatures = 9000 )
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()