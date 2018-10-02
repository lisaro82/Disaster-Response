import sys

import pandas as pd
import numpy as np

from sqlalchemy import create_engine, MetaData, Table


def load_data(p_messages_filepath, p_categories_filepath):
    
    # load messages dataset
    v_messages   = pd.read_csv(p_messages_filepath)
    
    # load categories dataset
    v_categories = pd.read_csv(p_categories_filepath)
    
    # Split categories into separate category columns
    v_categCols = v_categories['categories'].str.split(pat = ';', expand = True)
    
    # Use the first row of categories dataframe to create column names for the categories data
    v_categCols.columns = v_categCols.iloc[0].apply(lambda x: x[:-2])
    
    # Convert category values to just numbers 0 or 1
    v_categCols = v_categCols.apply(lambda x: pd.Series([item[-1] for item in x])).astype(int)
    
    v_categories = v_categories[['id']].merge(v_categCols, how = 'inner', left_index = True, right_index = True)
        
    return v_messages.merge(v_categories, how = 'left', on = 'id')


def clean_data(p_data):    
    # Remove duplicates by keeping only the first occurence into the index
    v_data  = p_data.reset_index()
    v_group = v_data.groupby('message').agg({'index': ['count', 'min', 'max']})
    v_group.columns = ['_'.join(item) for item in v_group.columns]
    
    v_data = v_data.merge(v_group[['index_min']], how = 'inner', left_on = 'index', right_on = 'index_min')
    v_data.drop(['index', 'index_min'], axis = 1, inplace = True)
    
    if len(v_data['message']) == len(v_data['message'].unique()):
        print(f"    Number of unique messages <<{len(v_data['message'].unique())}>>")
    else:
        raise Exception('Could not remove duplicates from the dataset.')

    return v_data


def save_data(p_data, p_database_filename):
    v_engine = create_engine(f'sqlite:///{p_database_filename}')
    
    # Create MetaData instance
    v_metadata = MetaData(v_engine, reflect = True)
    if 'tabMessages' in v_metadata.tables.keys():
        v_table = Table('tabMessages', v_metadata)
        v_table.drop(v_engine)
    
    p_data.to_sql('tabMessages', v_engine, index = False)
    return  


def main():
    if len(sys.argv) == 4:

        v_messages_filepath, v_categories_filepath, v_database_filepath = sys.argv[1:]

        print('Loading data...')
        print(f'    Messages   filepath <<{v_messages_filepath}>>')
        print(f'    Categories filepath <<{v_categories_filepath}>>')
        v_data = load_data(v_messages_filepath, v_categories_filepath)

        print('Cleaning data...')
        v_data = clean_data(v_data)
        
        print('Saving data...')
        print(f'    Database filepath <<{v_database_filepath}>>')
        save_data(v_data, v_database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('----------------------------------------------------------------------------------')
        print(' *** ERROR => Please provide the filepaths of the messages and categories datasets as the first and second argument respectively.')
        print('              Please provide the filepath of the database to save the cleaned data as the third argument.')
        print('              Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()