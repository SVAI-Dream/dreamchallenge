import os
import pandas as pd
obs_files = [f for f in '~/dream_data/']
ground_truth_data = pd.read_csv('~/dream_data/data_true.txt', sep='\t')

genes = ground_truth_data['Gene_ID'].values # list of gene names
by_genes = ground_truth_data.set_index('Gene_ID').T # expression matrix transposed
gene = genes[0] # first gene
y = by_genes[gene].as_matrix() # expression vector of a gene in 80 samples
ycl = by_genes[gene].apply(lambda x: int(x > 0)).as_matrix() # class tags of the expression vector
X = by_genes.drop([gene], axis=1).as_matrix() # take out the expression vector of the protein

# low (y = 0) vs. normal (y = 1) Classification with RFC + SMOTETomek
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support
from imblearn.combine import SMOTETomek, SMOTEENN
import pickle, gzip
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def rfc_with_smote(X, ycl, n_estimators=32, min_samples_leaf=5, max_depth=3):
    try:
        sm = SMOTETomek(k=5) # k is number of nearest neighbour
        X_smt, y_smt = sm.fit_sample(X, ycl) # returns re-sampled matrix and re-sampled label vector
    except:
        pass
    try:
        sm = SMOTETomek(k=2)
        X_smt, y_smt = sm.fit_sample(X, ycl)
    except:
        X_smt, y_smt = X, ycl

    X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.20, random_state=42)
    ycl_train = (y_train > 0).astype(int)
    ycl_test = (y_test > 0).astype(int)

    sfm = SelectFromModel(RandomForestClassifier(
        n_estimators = n_estimators,
        min_samples_leaf = min_samples_leaf,
        max_depth = max_depth
    )) # select features

    sfm.fit(X_train, ycl_train) # fit the model using training set
    X_train = sfm.transform(X_train) # reduce the matrix into selected features
    X_test = sfm.transform(X_test)

    rfc = RandomForestClassifier(
        n_estimators = n_estimators,
        min_samples_leaf = min_samples_leaf,
        max_depth = max_depth
    )

    rfc.fit_transform(X_train, ycl_train)
    precision, recall, f1, support = evaluate_model(rfc, X_test, ycl_test, threshold=0.5) # return some performance statistics of the model

    return rfc, precision, recall, f1, support

def save_model(model, model_name, model_output_path='~/dreamchallenge/sub1/RFC/model'):
    if not os.path.isdir(model_output_path):
        os.makedirs(model_output_path)

    with gzip.open(os.path.join(model_output_path, model_name + '.pkl.gz'), 'wb') as fm:
        pickle.dump(model, fm)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    pred = pd.DataFrame(model.predict_proba(X_test), columns=['Prob_low', 'Prob_normal'])
    pred_df = pd.concat([
        pd.DataFrame(y_test, columns=['grount_truth_class']),
        pred], axis=1
    )
    y_pred = pred_df['Prob_normal'].apply(lambda x: x > threshold)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    return precision, recall, f1, support

# Train RFC model gene by gene
from collections import namedtuple
results_cols= ['model_name', 'precision', 'recall','f1','support']
RFCResult = namedtuple('RFCResult', ' '.join(results_cols))
should_save_model=False
should_save_results=True
RFCResults = []

for gene in genes:
    model_name = 'RFC_{gene_name}'.format(gene_name=gene)
    print("Training RFC for gene: {}".format(model_name))
    y = by_genes[gene].as_matrix() # the protein's expression vector in 80 samples
    ycl = by_genes[gene].apply(lambda x: int(x > 0)).as_matrix() # the protein's class vector in 80 samples
    X = by_genes.drop([gene], axis=1).as_matrix() # take this protein out from the exp matrix

    model, precision, recall, f1, support = rfc_with_smote(X, ycl)

    RFCResults.append(RFCResult(
        model_name = model_name,
        precision = precision,
        recall = recall,
        f1 = f1,
        support = support
    ))

    if should_save_model:
        save_model(model, model_name)

if should_save_results:
    result_output_path = '~/dreamchallenge/sub1/RFC/summary/'
    if not os.path.isdir(result_output_path):
        os.makedirs(result_output_path)
    pd.DataFrame.from_records(RFCResults, columns=results_cols).to_csv(
        os.path.join(result_output_path, 'model_performances.csv')
        , index=False)
