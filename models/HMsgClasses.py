import re
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, recall_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

def messageTokenize(p_text):
    v_text = p_text
    
    v_url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    v_urls = re.findall(v_url_regex, v_text)
    
    for url in v_urls:
        v_text = v_text.replace(url, "urlplaceholder")
    
    sentence_list = nltk.sent_tokenize(v_text)
    v_first_verb = 0
    v_last_verb  = 0
    v_first_nnp  = 0
    v_last_nnp   = 0
    for sentence in sentence_list:
        pos_tags = nltk.pos_tag(word_tokenize(sentence))
        if v_first_verb == 0:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP', 'VBZ', 'VBG']:
                v_first_verb = 1
                
        if v_last_verb == 0:
            last_word, last_tag = pos_tags[-1]
            if last_tag in ['VB', 'VBP', 'VBZ', 'VBG']:
                v_last_verb = 1
                
        if v_first_nnp == 0:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['NNP']:
                v_first_nnp = 1
                
        if v_last_nnp == 0:
            last_word, last_tag = pos_tags[-1]
            if last_tag in ['NNP']:
                v_last_nnp = 1
    
    v_text = re.sub(r'[^a-zA-Z0-9]', ' ', v_text.lower())
    v_tokens = [item.strip() for item in word_tokenize(v_text) if item not in stopwords.words('english')]
    
    v_clean_tokens = []
    for token in v_tokens:
        token = WordNetLemmatizer().lemmatize(token)
        token = WordNetLemmatizer().lemmatize(token, pos = 'v')
        token = PorterStemmer().stem(token)
        v_clean_tokens.append(token.strip())
    
    v_text = ' '.join(v_clean_tokens)
    
    return (v_text, v_first_verb, v_last_verb, v_first_nnp, v_last_nnp)

def getTokenizedMessage(p_message):
    v_token = messageTokenize(v_data.loc[idx, 'message'])
    v_data = pd.DataFrame({ 'messageTokenized': v_token[0],
                            'flag_first_verb':  v_token[1],
                            'flag_last_verb':   v_token[2],
                            'flag_first_nnp':   v_token[3],
                            'flag_last_nnp':    v_token[4] })
    return v_data

#----------------------------------------------------------------------------------------------    
class HMsgExtractMessage():

    def transform(self, p_X):
        if type(p_X) == pd.core.frame.DataFrame:
            if 'messageTokenized' in p_X.keys():
                return p_X['messageTokenized']
            else:                
                for idx in p_X.index:
                    v_token = getTokenizedMessage(p_X.loc[idx, 'message'])
                    v_cols  = v_token.columns
                    p_X.loc[idx, v_cols] = v_token.iloc[0, v_cols]
                return p_X        
        return getTokenizedMessage(p_x)                

    def fit_transform(self, p_X, p_y = None):
        return self.transform(p_X)
    
    
#----------------------------------------------------------------------------------------------    
class HMsgCountVectorizer():
    
    __vectorize = CountVectorizer()
    
    def displayTop(self, p_X, p_top):
        v_reverse_dic = {}
        for key in self.__vectorize.vocabulary_:
            v_reverse_dic[self.__vectorize.vocabulary_[key]] = key        
        v_top = np.asarray(np.argsort(np.sum(p_X, axis=0))[0, (-1 * p_top):][0, ::-1]).flatten()
        
        print([v_reverse_dic[v] for v in v_top])        
        return
    
    def fit(self, p_X):
        self.__vectorize.fit(p_X)
        
    def transform(self, p_X):
        return self.__vectorize.transform(p_X)
        
    def fit_transform(self, p_X, p_y = None):
        self.fit(p_X)
        v_X = self.transform(p_X)
        print(f'------------------------------------------------------------------')
        print(f'Top 100 words are the following: ')
        self.displayTop(p_X = v_X, p_top = 100)
        return v_X
    
    
#----------------------------------------------------------------------------------------------    
class HMsgTfidfTransformer():
    
    __transformer = TfidfTransformer()
    
    def fit(self, p_X):
        self.__transformer.fit(p_X)
        
    def transform(self, p_X):
        v_X = self.__transformer.transform(p_X)
        v_X.sort_indices()
        return v_X
        
    def fit_transform(self, p_X, p_y = None):
        self.fit(p_X)
        return self.transform(p_X)
        
        
#----------------------------------------------------------------------------------------------    
class HMsgFeatureExtract():
    
    __column = ''
    
    def __init__(self, p_column):
        self.__column = p_column
        
    def transform(self, p_X):
        return p_X[self.__column].values.reshape(-1, 1)
        
    def fit_transform(self, p_X, p_y = None):
        return self.transform(p_X)
        
        
#----------------------------------------------------------------------------------------------    
class HMsgClassifier():
    __models   = []
    __classes  = {}
    __mapGenre = None
    
    def setGenreMap(self, p_mapGenre):
        self.__mapGenre = p_mapGenre
        
    def tuneHyperparams(self, p_X_train, p_y_train, p_className):        
        print(f'    Tuning hyper-parameters for class: <<{p_className}>> (Recall).')  
        v_param_grid = { 'C': [0.01, 0.1, 1, 10] }  
        v_model = LinearSVC( random_state      = 42,
                             class_weight      = 'balanced' )
        
        v_param_grid = { 'loss':    ['hinge', 'squared_hinge', 'perceptron', 'squared_loss'],
                         'penalty': ['l1', 'l2', 'elasticnet'],
                         'alpha':   [0.0001 * item for item in range(1, 10, 3)] }  
        v_model = SGDClassifier( random_state  = 42,
                                 class_weight  = 'balanced',
                                 max_iter      = 3000,
                                 tol           = 1e-3 )
        
        v_param_grid = { #'kernel':    ['linear', 'poly', 'rbf', 'sigmoid'],
                         'C':         [0.01, 0.1, 1, 10],
                         'degree':    range(1, 5) }  
        v_model = SVC( random_state  = 42,
                       class_weight  = 'balanced',
                       kernel        = 'poly' )
        
        v_score = make_scorer(recall_score, average = 'macro')
        v_grid_search = GridSearchCV( v_model, 
                                      param_grid = v_param_grid, 
                                      cv = 6, 
                                      scoring = v_score,
                                      return_train_score = True,
                                      verbose = 10 )
        v_grid_search.fit(p_X_train, p_y_train)
        
        v_data = pd.DataFrame({ '1. params':            pd.Series(v_grid_search.cv_results_["params"]),
                                '2. mean_train_score':  pd.Series(v_grid_search.cv_results_["mean_train_score"]),
                                '3. mean_test_score':   pd.Series(v_grid_search.cv_results_["mean_test_score"]),
                                '4. std_train_score':   pd.Series(v_grid_search.cv_results_["std_train_score"]),
                                '5. std_test_score':    pd.Series(v_grid_search.cv_results_["std_test_score"]) })
        print(v_data)
        
        return v_grid_search.best_estimator_
    
    def fit(self, p_X, p_y):
        v_classes = p_y.columns.tolist()
        for idx in range(p_y.shape[1]):
            print(f'    Fit model for class: <<{v_classes[idx]}>>.')             
            self.__models.append(self.tuneHyperparams(p_X, p_y.iloc[:, idx], v_classes[idx]))
            self.__classes[v_classes[idx]] = idx
    
    def predict(self, p_X):        
        v_return = np.zeros([p_X.shape[0], len(self.__models)])
        for idx in range(len(self.__models)):
            v_return[:, idx] = self.__models[idx].predict(p_X)
        return v_return
    
    def classificationReport(self, p_y_true, p_y_pred, p_classes, p_showSummary = True): 
        if p_showSummary:
            v_data = pd.DataFrame()
            for key, value in self.__classes.items():
                v_score = precision_recall_fscore_support(p_y_true.iloc[:, value], p_y_pred[:, value], average = "weighted")
                v_data = pd.concat([ v_data, pd.DataFrame({ '__True Sum':  int(p_y_true.iloc[:, value].sum()),
                                                            '__Pred Sum':  int(p_y_pred[:, value].sum()),
                                                            'Precision':   round(v_score[0], 4),
                                                            'Recall':      round(v_score[1], 4),
                                                            'F-score':     round(v_score[2], 4) }, index = [key]) ])
            print('\n-------------------------------------------------------------------')
            print(v_data)
            print(' ')
                        
        for className in p_classes:
            v_idx = self.__classes[className]
            print('\n-------------------------------------------------------------------')
            print(f'Details for class: <<{className}>>.')     
            
            print(f'Classification Report:')     
            print(classification_report(p_y_true.iloc[:, v_idx], p_y_pred[:, v_idx]))
            
            print(f'Confusion Matrix:')     
            print(confusion_matrix(p_y_true.iloc[:, v_idx], p_y_pred[:, v_idx]))
            print(' ')
        return