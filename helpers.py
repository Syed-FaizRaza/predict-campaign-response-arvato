import pandas as pd, numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RepeatedStratifiedKFold as Val
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class CleanData:
    """Preprocess data: Find missing values, delete categorize, add features
    
    Returns:
        [type] -- [description]
    """
    def __init__(self):
        self.values_for_missing = None
        self.del_cols = ['KBA13_HERST_SONST', #dup column with KBA13_FAB_SONSTIGE
                         'LP_FAMILIE_FEIN',#dup info with LP_FAMILIE_GROB
                         'CAMEO_INTL_2015', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB',#dup info by 2 vars
                         # 'HAUSHALTSSTRUKTUR',#dup info by mul vars, not in dataset
                         'ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', #lots of missing
                         'KK_KUNDENTYP', 'EXTSEL992',#lots of missing
                        ]
        self.categorical_cols = ['AGER_TYP', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'D19_KONSUMTYP',
                                 'CJT_GESAMTTYP','ZABEOTYP', 'GEBAEUDETYP_RASTER', 'GFK_URLAUBERTYP',
                                 'HEALTH_TYP', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB',
                                 'NATIONALITAET_KZ', 'PRAEGENDE_JUGENDJAHRE', 'RETOURTYP_BK_S', 'TITEL_KZ',
                                 # 'GEBAEUDETYP', not in dataset
                                 'D19_LETZTER_KAUF_BRANCHE',
                                ]
        self.label_dict = None

    def find_missing(self, df):
        if not self.values_for_missing:
            attr_df = pd.read_csv('input/attribute_values.csv').set_index('attribute')
            self.values_for_missing = attr_df.value[attr_df.meaning.str.contains('unknown')].to_dict()
        try:
            for col in df.columns:
                missing_vals = [int(v) for v in self.values_for_missing[col].split(', ')]
                df[col] = np.where(df[col].isin(missing_vals), None, df[col])
        except:
            pass
        df['CAMEO_DEUG_2015'] = np.where(df['CAMEO_DEUG_2015']=='X', None, df['CAMEO_DEUG_2015']) 
        df['CAMEO_DEU_2015'] = np.where(df['CAMEO_DEU_2015']=='X', None, df['CAMEO_DEUG_2015']) 
        # df['CAMEO_INTL_2015'] = np.where(df['CAMEO_INTL_2015']=='XX', None, df['CAMEO_DEUG_2015']) 
        return df

    def remap_values(self, df):
        def map_int(x):
            if pd.isnull(x):
                return None
            return int(x)
        
        df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(map_int)
        # df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].apply(map_int)
        df['LP_FAMILIE_GROB'] = df['LP_FAMILIE_GROB']\
                            .map({1:1, 2:2, 3:3, 4:3, 5:3, 6:4, 7:4, 8:4, 9:5, 10:5, 11:5})
        df['LP_STATUS_GROB'] = df['LP_STATUS_GROB']\
                            .map({1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:5})
        df['OST_WEST_KZ'] = df['OST_WEST_KZ'] == 'W'                    
        df['BUILIDING_TYPE'] = df['GEBAEUDETYP'] //2
        df['W_WOUT'] = df['GEBAEUDETYP'] % 2
        df['EINGEFUEGT_AM'] = df['EINGEFUEGT_AM'].str[:4].apply(map_int)
        df = df.drop(columns=['GEBAEUDETYP'])

        return df
    
    def encode_labels(self, df):
        if self.label_dict:
            for k in self.label_dict:
                df[k] = self.label_dict[k].transform(df[k].fillna('nan').astype(str))
            return df

        label_dict = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('nan').astype(str))
            label_dict[col] = le
        self.label_dict = label_dict
        return df

    def create_time_difference_variables(self, df):
        time_cols = ['GEBURTSJAHR', 'EINGEZOGENAM_HH_JAHR','EINGEFUEGT_AM', 'MIN_GEBAEUDEJAHR']
        for i in range(len(time_cols)-1):
            for j in range(i+1, len(time_cols)):
                df['diff_'+time_cols[i]+time_cols[j]] = df[time_cols[i]] - df[time_cols[j]]
        return df

    def clean(self, df, label_encode=False):
        if 'LNR' in df:
            df = df.set_index('LNR')
        df = self.find_missing(df)
        df = self.remap_values(df)
        df = self.create_time_difference_variables(df)
        df = df.drop(columns=self.del_cols)
        df['sum_null'] = df.isnull().sum(axis=1)
        if label_encode:
            return self.encode_labels(df)
        return df



class KFoldCrossVal:
    def __init__(self, n_splits=5, n_repeats=10):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.best_features = None


    def fit(self, df, clf, features=None, cat_features=None, top_n=20, get_best_features=True):
        skf = Val(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=7)
        if not features:
            features = list(df.columns)
            features.remove('RESPONSE')
        X = df[features]
        y = df.RESPONSE
        best_auc = 0
        self.train_evals = []
        self.test_evals = []
        print(get_best_features)
        if get_best_features:
            self.best_features = Counter()
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if clf.__module__.startswith('sklearn'):
                clf.fit(X_train, y_train)
                train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
                test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            else:
                clf.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        eval_metric='auc',
                        verbose=True
                )
                train_auc = clf.evals_result_['validation_0']['auc'][0]
                test_auc = clf.evals_result_['validation_1']['auc'][0]
            if test_auc>best_auc:
                self.best_est = clf
                best_auc = test_auc
                
            self.train_evals.append(train_auc)
            self.test_evals.append(test_auc)

            if get_best_features:
                t = pd.Series(clf.feature_importances_, index=X_train.columns)
                for f in t.sort_values(ascending=False)[:top_n].index:
                    self.best_features[f]+=1
        print('Train AUC:', np.mean(self.train_evals), '  std:', np.std(self.train_evals))
        print('Test AUC:', np.mean(self.test_evals), '  std:', np.std(self.test_evals))

    def fit_downsample(self, df, clf, features=None, frac=0.1):
        df_yes = df[df.RESPONSE==1]
        df_no = df[df.RESPONSE==0]

        if not features:
            features = df_no.drop(columns=['RESPONSE']).columns
        df_no = df_no.sample(frac=1, random_state=7)
        n = int(0.05*len(df_no))

        for i in range(0, len(df_no), n):
            df = pd.concat((df_yes, df_no.iloc[i:(i+n)])).sample(frac=1, random_state=7)
            pass





def find_best_match(df1, df2, thres=.8):
    most_diff = Counter()
    df2 = df2.fillna(-1)
    def get_most_similar(x):
        t = (x.fillna(-1) == df2).mean(axis=1)
        lnr = t.argmax()
        if t[lnr] > thres:
            for col in x.index:
                if (x[col]!=-1)&(df2[col][lnr]!=-1)&(x[col]!=df2[col][lnr]):
                    most_diff[col]+=1
    best_match_w_yes = df1.apply(get_most_similar, axis=1)
    return best_match_w_yes, most_diff


def feature_generation(df):
    col_list = list(df)
    col_list.remove('RESPONSE')
    for c in col_list:
        col_list.remove(c)
        t = df[col_list].corrwith(df[c])
        t
    return df


def feature_selection(feature_matrix:pd.DataFrame, missing_threshold=90, correlation_threshold=0.95):
    """Feature selection for a dataframe."""
    bool_cols = feature_matrix.select_dtypes(bool).columns
    feature_matrix[bool_cols] = feature_matrix[bool_cols].astype(int)
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    _, idx = np.unique(feature_matrix, axis = 1, return_index = True)
    feature_matrix = feature_matrix.iloc[:, idx]
    n_non_unique_columns = n_features_start - feature_matrix.shape[1]
    print('{}  non-unique valued columns.'.format(n_non_unique_columns))

    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                                        missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_non_unique_columns + n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix

class Viz():
    def __init__(self, trained_clf):
        self.trained_clf = trained_clf

    def plot_learning_curve(self):
        results = self.trained_clf.evals_result()
        epochs = len(results['validation_0']['auc'])
        x_axis = range(0, epochs)
        
        plt.plot(x_axis, results['validation_0']['auc'], label='Train')
        plt.plot(x_axis, results['validation_1']['auc'], label='Test')
        plt.ylabel('Auc')
        plt.show()

if __name__ == '__main__':
    pass
        

    
# filter num of features (1) 76.3
# reduced feature set 77.2
# match and select best, integrate to (1) 77.3