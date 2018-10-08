import sys

import pandas as pd
import numpy as np

import gc

from sqlalchemy import create_engine, MetaData, Table

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib

from HMsgClasses import getTokenizedMessage
from HMsgClasses import HMsgExtractMessage
from HMsgClasses import HMsgCountVectorizer
from HMsgClasses import HMsgTfidfTransformer
from HMsgClasses import HMsgFeatureExtract
from HMsgClasses import HMsgClassifier

from IPython.display import display

def load_data(p_database_filepath, p_reloadData = False):  

    v_cols = [ 'messageTokenized', 
               'flag_first_verb', 'flag_last_verb', 
               'flag_first_nnp', 'flag_last_nnp', 'flag_nnp' ]         
    
    v_engine = create_engine(f'sqlite:///{p_database_filepath}', echo = False)
    v_metadata = MetaData()
    v_metadata.reflect(bind = v_engine)

    v_tabName = 'tabMessagesTokenized'
    if ( p_reloadData 
         or v_tabName not in v_metadata.tables.keys() ):

        v_data = pd.read_sql_table('tabMessages', v_engine)

        for item in v_cols:
            v_data[item] = np.NaN

        v_count = 0
        for idx in v_data.index:
            v_token = getTokenizedMessage(v_data.loc[idx, 'message'])
            v_data.loc[idx, v_cols] = v_token.loc[0, v_cols]

            v_count += 1
            if v_count % 300 == 0: print(f'Rows processed: {v_count} / {v_data.shape[0]}')                            
      
        print('    All rows processed.')  
        v_data.to_sql(v_tabName, v_engine, index = False, if_exists = 'replace', index_label = 'id')        
        print('    Processed data saved.')  
        
    v_data = pd.read_sql_table(v_tabName, v_engine) 
    
    del v_engine
    gc.collect()
    
    print('    Columns with null values.') 
    v_null = v_data.isnull().sum()
    display(v_null[v_null != 0])
    
    v_target = v_data.copy()
    v_target.drop(v_cols, axis = 1, inplace = True)
    v_target.drop(['id', 'message', 'original'], axis = 1, inplace = True) 
    print('    Target created.')  
    
    v_mapGenre = { 'news':    0,
                   'direct':  1,
                   'social':  2 }
    v_target['genre'] = v_target['genre'].map(v_mapGenre)  
    print('    Target replace genre column.')  
    
    # We check that all the classes have at least one value specified for at least 2 classes. If only one class is present
    # than we remove the category from the prediction list.
    v_sum = v_target.sum(axis = 0)
    v_target.drop(v_sum[v_sum == 0].index, axis = 1, inplace = True)
    
    v_cols.append('message')
    return v_data[v_cols], v_target


def build_model(p_CVSplits, p_pointsBin, p_maxCateg, p_maxFeatures, p_debug):
    return Pipeline([ ('features', FeatureUnion([ ('text_pipeline', Pipeline([ ('step_01', HMsgExtractMessage()),
                                                                               ('step_02', HMsgCountVectorizer( max_features = p_maxFeatures )),
                                                                               ('step_03', HMsgTfidfTransformer()) 
                                                                             ]) ),
                                                  ('feat_01', HMsgFeatureExtract('flag_first_verb')),
                                                  ('feat_02', HMsgFeatureExtract('flag_last_verb')),
                                                  ('feat_03', HMsgFeatureExtract('flag_first_nnp')),
                                                  ('feat_04', HMsgFeatureExtract('flag_last_nnp')),
                                                  ('feat_05', HMsgFeatureExtract('flag_nnp')),
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
    return

def executeMain( p_database_filepath, 
                 p_model_filepath, 
                 p_CVSplits     = 12, 
                 p_pointsBin    = 15, 
                 p_maxCateg     = None, 
                 p_maxFeatures  = 12000, 
                 p_reloadData   = False, 
                 p_debug        = False ):
    print('Loading data.')
    print(f'    Database filepath <<{p_database_filepath}>>')        
    v_X, v_y = load_data(p_database_filepath, p_reloadData)
    print('    Data loaded.')  
    
    # Create a testing dataset
    X_train, X_test, y_train, y_test = train_test_split(v_X, v_y, test_size = 0.20, random_state = 42)
    print('Testing dataset created.')

    # Create a validation dataset
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.50, random_state = 42)
    print('Validation dataset created.')
    
    print('Building model.')
    v_model = build_model(p_CVSplits, p_pointsBin, p_maxCateg, p_maxFeatures, p_debug)
    
    print('Check fit_transform method on train data.')
    v_train_01 = v_model.named_steps['features'].fit_transform(X_train)
    
    print('Check transform method on train data.')    
    v_train_02 = v_model.named_steps['features'].transform(X_train)    
    assert (z in v_train_01 for z in v_train_02)
    
    print('Transform validation dataset.')
    X_valid = v_model.named_steps['features'].transform(X_valid)

    print('Training model.')
    v_model.fit(X_train, (y_train, X_valid, y_valid))
    
    print('Evaluating model.')
    y_pred_test = evaluate_model(v_model, X_test, y_test, None)

    print('Saving model.')
    print(f'    Model filepath <<{p_model_filepath}>>')
    save_model(v_model, p_model_filepath)

    print('Trained model saved!')
    return v_model, X_train, X_test, y_train, y_test, y_pred_test 

def runMain():
    if len(sys.argv) == 3:
        v_database_filepath, v_model_filepath = sys.argv[1:]
        _ = executeMain( p_database_filepath = v_database_filepath, 
                         p_model_filepath    = v_model_filepath,
                         p_CVSplits          = 3,
                         p_pointsBin         = 5,
                         p_maxFeatures       = 12000,
                         p_reloadData        = True )
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    runMain()