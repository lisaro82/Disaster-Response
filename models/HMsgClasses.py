import re
import pandas as pd
import numpy as np

# Try to import the libraries for nltk. In case of error, download the necessary components and try again.
for _ in range(2):
    try:
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.stem.porter import PorterStemmer        
        v_words = stopwords.words('english')
        break
    except:
        import nltk
        nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, recall_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.svm import LinearSVC

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

def getTokenizedMessage(p_message, p_data = None):
    if not p_data is None:
        v_token = messageTokenize(p_data.loc[idx, 'message'])
    else:
        v_token = messageTokenize(p_message)
        
    v_data = pd.DataFrame({ 'messageTokenized': v_token[0],
                            'flag_first_verb':  v_token[1],
                            'flag_last_verb':   v_token[2],
                            'flag_first_nnp':   v_token[3],
                            'flag_last_nnp':    v_token[4] }, index = [0])
                            
    return v_data

#----------------------------------------------------------------------------------------------    
class HMsgExtractMessage():

    def transform(self, p_X):
        if type(p_X) == pd.core.frame.DataFrame:
            if 'messageTokenized' in p_X.keys():
                return p_X['messageTokenized']
            else:                
                for idx in p_X.index:
                    v_token = getTokenizedMessage(p_X.loc[idx, 'message'], p_X)
                    v_cols  = v_token.columns
                    p_X.loc[idx, v_cols] = v_token.iloc[0, v_cols]
                return p_X        
        return getTokenizedMessage(p_x)                

    def fit_transform(self, p_X, p_y = None):
        return self.transform(p_X)
    
    
#----------------------------------------------------------------------------------------------    
class HMsgCountVectorizer(CountVectorizer):
    
    def displayTop(self, p_X, p_top):
        v_reverse_dic = {}
        for key in self.vocabulary_:
            v_reverse_dic[self.vocabulary_[key]] = key        
        v_top = np.asarray(np.argsort(np.sum(p_X, axis=0))[0, (-1 * p_top):][0, ::-1]).flatten()
        
        print([v_reverse_dic[v] for v in v_top])        
        return
    
    def fit(self, p_X):  
        super(HMsgCountVectorizer, self).fit(p_X)
        
    def transform(self, p_X):
        return super(HMsgCountVectorizer, self).transform(p_X)
        
    def fit_transform(self, p_X, p_y = None):
        v_X = super(HMsgCountVectorizer, self).fit_transform(p_X)
        print(f'------------------------------------------------------------------')
        print(f'Top 100 words are the following: ')
        self.displayTop(p_X = v_X, p_top = 100)
        return v_X
    
    
#----------------------------------------------------------------------------------------------    
class HMsgTfidfTransformer(TfidfTransformer):
    
    __transformer = TfidfTransformer()
    
    def fit(self, p_X):
        super(HMsgTfidfTransformer, self).fit(p_X)
        
    def transform(self, p_X):
        v_X = super(HMsgTfidfTransformer, self).transform(p_X)
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
class HMsgFeatureUnion():
    
    __data = None
    
    def __init__(self, p_data):
        self.__data = p_data
    
    def transform(self, p_X):
        return self.__data
    
    def fit_transform(self, p_X, p_y = None):
        return self.transform(p_X)
    
        
#----------------------------------------------------------------------------------------------    
class HMsgClassifier():
    
    __debug      = None
    __CVSplits   = None
    __pointsBin  = None  
    __maxCateg   = None
    __models     = []
    __classes    = {}
    
    def __init__(self, p_CVSplits = 12, p_pointsBin = 15, p_maxCateg = None, p_debug = False):
        self.__CVSplits  = p_CVSplits
        self.__pointsBin = p_pointsBin
        self.__maxCateg  = p_maxCateg
        self.__debug     = p_debug
        return
        
    def tuneHyperparams(self, p_X_train, p_y_train, p_className):   
        def gridSearch(p_run, p_model, p_weight, p_param_grid, p_verbose):
            v_param_grid = p_param_grid
            if 'C' in v_param_grid.keys():
                v_param_grid['C'] = np.round(v_param_grid['C'], 4).tolist()
            print(f'       {p_run}. Grid Search model <<{p_model}>>; weight: <<{p_weight}>>; parameters: <<{v_param_grid}>>')
            v_model = p_model( random_state      = 42,
                               class_weight      = p_weight )

            v_score = make_scorer(recall_score, average = 'macro')
            v_grid_search = GridSearchCV( v_model, 
                                          param_grid = p_param_grid, 
                                          cv = self.__CVSplits, 
                                          scoring = v_score,
                                          return_train_score = True,
                                          verbose = p_verbose )
            
            v_grid_search.fit(p_X_train, p_y_train)
            
            if self.__debug:
                v_data = pd.DataFrame({ 'Params':      pd.Series(v_grid_search.cv_results_['params']),
                                        'Mean Train':  pd.Series(v_grid_search.cv_results_['mean_train_score']),
                                        'Mean Test':   pd.Series(v_grid_search.cv_results_['mean_test_score']),
                                        'Std Train':   pd.Series(v_grid_search.cv_results_['std_train_score']),
                                        'Std Test':    pd.Series(v_grid_search.cv_results_['std_test_score']) })
                v_cols = v_data.drop('Params', axis = 1).columns
                v_data[v_cols] = v_data[v_cols].round(4)
                print(v_data[['Params', 'Mean Train', 'Mean Test', 'Std Train', 'Std Test']])
            
            print(f'       Best parameters selected: <<{v_grid_search.best_params_}>> for score <<{v_grid_search.best_score_}>>.')
            return v_grid_search
        
        def runLinearSVC(p_run, p_weight, p_minScore = 0.85):
            v_run = p_run
            v_grid_search = gridSearch( p_run        = v_run,
                                        p_model      = LinearSVC,
                                        p_weight     = p_weight,
                                        p_param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10] },
                                        p_verbose    = 1 )
            v_C = v_grid_search.best_params_['C']                    
            for _ in range(10):
                v_run += 1
                v_params      = v_grid_search.best_params_            
                v_list = np.linspace( v_params['C'] / 3 if v_params['C'] / 3 > 1e-6 else 1e-6, 
                                      v_params['C'], 
                                      self.__pointsBin ).tolist()
                v_list.extend(np.linspace(v_params['C'], v_params['C'] * 3, self.__pointsBin).tolist())
                v_params['C'] = sorted(set(v_list))
                v_grid_search = gridSearch( p_run        = v_run,
                                            p_model      = LinearSVC,
                                            p_weight     = p_weight,
                                            p_param_grid = v_params,
                                            p_verbose    = 1 )

                v_best_model = v_grid_search 

                # If we have the same score twice, then stop processing, as we have our best model
                if v_best_model.best_params_['C'] == v_C:
                    break
                else:
                    v_C = v_grid_search.best_params_['C']
                    
            return v_best_model, False if v_best_model.best_score_ < p_minScore else True, v_run
        
        #-------------------------------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------------------------------
        print(f'    Tuning hyper-parameters for class: <<{p_className}>> (Recall).')  
        
        #-------------------------------------------------------------------------------------------------------------------------
        # The first model that we will try is the LinearSVC model with balanced classes. If the final score is not satisfying, than we will
        # try the LinearSVC model with custom weights.
        v_run = 1
        v_best_model, v_found, v_run = runLinearSVC(v_run, 'balanced')
        
        if not v_found:
            v_values = p_y_train.value_counts()
            v_weight  = (1 / (v_values / v_values.iloc[0])).apply(lambda x: 1 if x == 1 else x * 2 if x * 2 < 100 else x).to_dict()
            v_grid_search, v_found, v_run = runLinearSVC(v_run, v_weight)
            
            if v_grid_search.best_score_ > v_best_model.best_score_ + 0.02: # Boost the score for best model with "balanced" weight in   
                                                                            # order to make it the prefered one, unless there is a  
                                                                            # difference bigger than 0.02.
                v_best_model = v_grid_search
        
        print(f'       Final parameters selected: <<{v_best_model.best_params_}>> for score <<{v_best_model.best_score_}>>.')
        
        return v_best_model
    
    def fit(self, p_X, p_y):                    
        
        v_classes = p_y.columns.tolist()
        X_data = p_X
        y_data = p_y
        
        v_count = 0
        v_enriched = False # Used to flag that the dataset has been enriched with new categories which are well predicted
        v_range = self.__maxCateg if not self.__maxCateg is None else p_y.shape[1]
        for idx in range(v_range):
            v_key = v_classes[idx]
            
            X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size = 0.10, random_state = 42)
            
            print('\n-------------------------------------------------------------------')
            v_count += 1
            print(f'    {v_count}. Fit model for class: <<{v_classes[idx]}>> ({X_train.shape}).')    
            v_model = self.tuneHyperparams(X_train, y_train.iloc[:, idx], v_classes[idx])   
            
            y_true = y_valid.iloc[:, idx]
            y_pred = v_model.predict(X_valid)
            
            v_score_Ma = recall_score(y_true, y_pred, average = "macro")
            v_score_We = recall_score(y_true, y_pred, average = "weighted")            
            print(f'\n          *** Score on validation dataset (macro/weighted) <<{v_score_Ma}>> / <<{v_score_We}>>.')
            print(confusion_matrix(y_true, y_pred))
            
            self.__models.append({ 'key':                    v_key,
                                   'model':                  v_model.best_estimator_,
                                   'model_bestScore':        v_model.best_score_,
                                   'valid_score_recall_Ma':  v_score_Ma,
                                   'valid_score_recall_We':  v_score_We,
                                   'useModel':               True if v_model.best_score_ > 0.8 else False,
                                   'update':                 False })
            
            self.__classes[v_key] = { 'categ_idx':     idx,
                                      'model_idx':     idx,
                                      'createFeature': True if v_model.best_score_ > 0.9 else False }      
            if self.__classes[v_key]['createFeature']:
                v_enriched = True
                # Integrate the category column for the prediction of the later categories
                X_data = FeatureUnion([ ('feat_01', HMsgFeatureUnion(X_data)),
                                        ('feat_02', HMsgFeatureUnion(y_data.iloc[:, idx].values.reshape(-1, 1))) ]).fit_transform(None)
        
        if v_enriched:
            # If the model for a particular category is not ok, than try to generate a new model based on the enriched dataset
            for model in self.__models:
                if not model['useModel']:
                    v_key = model['key']
                    v_categ_idx = self.__classes[v_key]['categ_idx']

                    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size = 0.15, random_state = 42)

                    print('\n-------------------------------------------------------------------')
                    v_count += 1
                    print(f'    {v_count}. ReFit model for class: <<{v_key}>> ({X_train.shape}).')    
                    v_model = self.tuneHyperparams(X_train, y_train.iloc[:, v_categ_idx], v_key)  

                    y_true = y_valid.iloc[:, v_categ_idx]
                    y_pred = v_model.predict(X_valid)

                    v_score_Ma = model['valid_score_recall_Ma']
                    v_score_We = model['valid_score_recall_We']
                    print(f'\n          *** Prev Score on validation dataset (macro/weighted) <<{v_score_Ma}>> / <<{v_score_We}>>.')

                    v_score_Ma = recall_score(y_true, y_pred, average = "macro")
                    v_score_We = recall_score(y_true, y_pred, average = "weighted")
                    print(f'\n          *** Score on validation dataset (macro/weighted) <<{v_score_Ma}>> / <<{v_score_We}>>.')
                    print(confusion_matrix(y_true, y_pred))                

                    # If the new score is higher, than we chack that the gain on the validation dataset is also there. We consider
                    # that the gain / loss on weighted score is half as important
                    if ( v_model.best_score_ > model['model_bestScore']
                         and ( v_score_Ma - model['valid_score_recall_Ma']
                               + (v_score_We - model['valid_score_recall_We']) / 2 ) > 0 ):                
                        self.__models.append({ 'key':                    v_key,
                                               'model':                  v_model.best_estimator_,
                                               'model_bestScore':        v_model.best_score_,
                                               'valid_score_recall_Ma':  v_score_Ma,
                                               'valid_score_recall_We':  v_score_We,
                                               'useModel':               True,
                                               'update':                 True })
                        self.__classes[v_key]['model_idx'] = len(self.__models) - 1
                        print(f'\n          *** Model has been refitted from <<{model["model_bestScore"]}>> to <<{v_model.best_score_}>>.')
        return
    
    def generatePredictions(self, p_X, p_predictProba = False):        
        X_data = p_X
        v_return = np.zeros([p_X.shape[0], len(self.__classes)])
        for model in self.__models:
            if model['useModel']:
                v_key = model['key']
                v_categ_idx = self.__classes[v_key]['categ_idx']
                v_model_idx = self.__classes[v_key]['model_idx']
                v_model = self.__models[v_model_idx]['model']
                
                y_pred = v_model.predict(X_data)
                v_return[:, v_categ_idx] = y_pred
                
                if p_predictProba:
                    y_pred_proba = v_model.predict_proba(X_data)   
                    v_return[:, v_categ_idx] = y_pred_proba

                if self.__classes[v_key]['createFeature']:                                            
                    # Integrate the category column for the prediction of the later categories
                    X_data = FeatureUnion([ ('feat_01', HMsgFeatureUnion(X_data)),
                                            ('feat_02', HMsgFeatureUnion(y_pred.reshape(-1, 1))) ]).fit_transform(None)
        
        return v_return
    
    def predict(self, p_X):        
        return self.generatePredictions(p_X = p_X, p_predictProba = False)
    
    def predict_proba(self, p_X):         
        return self.generatePredictions(p_X = p_X, p_predictProba = True)
    
    def classificationReport(self, p_y_true, p_y_pred, p_classes, p_showSummary = True): 
        if p_showSummary:
            v_data = pd.DataFrame()
            for model in self.__models:
                if not model['update']: # If this is the first version of a model generated for the given category, than select 
                                        # correct category index and calculate the scores
                    v_key = model['key']
                    v_categ_idx = self.__classes[v_key]['categ_idx']
                    
                    y_true = p_y_true.iloc[:, v_categ_idx]
                    y_pred = p_y_pred[:, v_categ_idx]
                        
                    v_score = precision_recall_fscore_support(y_true, y_pred, average = "weighted")
                    v_score_recall = recall_score(y_true, y_pred, average = "macro")
                    v_data = pd.concat([ v_data, pd.DataFrame({ '__True Sum':  int(y_true.sum()),
                                                                '__Pred Sum':  int(y_pred.sum()),
                                                                'Precision':   round(v_score[0], 4),
                                                                'Recall_We':   round(v_score[1], 4),
                                                                'Recall_Ma':   round(v_score_recall, 4),
                                                                'F-score':     round(v_score[2], 4) }, index = [v_key]) ])
            
            v_data['__Diff'] = v_data['__True Sum'] - v_data['__Pred Sum']
            
            print('\n-------------------------------------------------------------------')
            print(v_data)
            print(' ')
                        
        if not p_classes is None:
            v_classes = p_classes
        else:
            v_classes = self.__classes.keys()
            
        for className in v_classes:
            v_idx = self.__classes[className]['categ_idx']
            print('\n-------------------------------------------------------------------')
            print(f'Details for class: <<{className}>>.')     
            
            print(f'Classification Report:')     
            print(classification_report(p_y_true.iloc[:, v_idx], p_y_pred[:, v_idx]))
            
            print(f'Confusion Matrix:')     
            print(confusion_matrix(p_y_true.iloc[:, v_idx], p_y_pred[:, v_idx]))
            print(' ')
        return