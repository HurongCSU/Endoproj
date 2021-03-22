import csv
import pandas as pd
import numpy as np
from tpot import TPOTClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import joblib

import csv
import pandas as pd
import numpy as np
from sklearn import model_selection
from tpot import TPOTClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
#from pyearth import Earth
from sklearn.cross_decomposition import PLSRegression


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skfeature.function.statistical_based import t_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF

from sklearn.feature_selection import mutual_info_classif
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import DISR

from scipy.stats import wilcoxon

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import warnings

num_fea = 10
name_str = '10_uterus.pkl'
output = open("endohand10_uterus.txt","w")
file = 'fea10_uterus.xls'

file_feature = "./csv/uterus_features.csv"
file_train = "./csv/train.csv"
file_validate = "./csv/validation.csv"
file_test = "./csv/test.csv"

f = open(file_feature)
csv_f = csv.reader(f)
features = next(csv_f)
dataset = pd.read_csv(file_feature, names=features, usecols=range(1,6098), dtype=np.float64, skiprows=1, low_memory=False)
f = open(file_train)
csv_f = csv.reader(f)
features = next(csv_f)
dataset_train = pd.read_csv(file_train, names=features, usecols=range(1,1), dtype=np.float64, skiprows=1, low_memory=False)

with open('./csv/train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    train_list = [row[1] for row in reader]
with open('./csv/validation.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    validation_list = [row['patient'] for row in reader]
with open('./csv/test.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    test_list = [row['patient'] for row in reader]

dataset['outcome'] = pd.to_numeric(dataset['outcome'],errors='coerce')
array_OG = dataset.values
print(array_OG.shape)
train_list = train_list[1:]
validation_list = validation_list[0:]
test_list = test_list[0:]
#print(test_list)
#print(train_list)
#print(validation_list)

def cat_str(num_list):
    n_list = []
    for i in num_list:
        temp = i[12:]
        n_list.append(temp)
    n_list = [int(x) for x in n_list]
    return n_list

train_list = cat_str(train_list)
validation_list = cat_str(validation_list)
test_list = cat_str(test_list)

#print(train_list)
#print(validation_list)
#print(test_list)
print(test_list)

train_feature = []
validate_feature = []
test_feature = []
count = 1
for i in range(len(array_OG)):
    num = i + 1
    if num in train_list:
        train_feature.append(array_OG[i])
    elif num in validation_list:
        validate_feature.append(array_OG[i])
    elif num in test_list:
        #print(count)
        count = count + 1
        test_feature.append(array_OG[i])
        #print(num)
        #print(array_OG[i,6096])
        
train_feature = np.array(train_feature)
validate_feature = np.array(validate_feature)
test_feature = np.array(test_feature)

train_feature = pd.DataFrame(train_feature)

train_feature.dropna(axis=1, thresh=2, inplace=True)
train_feature.dropna(how='all',thresh = 20,inplace=True)
train_feature = np.array(train_feature)
wh_inf = np.isinf(train_feature)
train_feature[wh_inf]=0
wh_nan = np.isnan(train_feature)
train_feature[wh_nan]=0

validate_feature = pd.DataFrame(validate_feature)
validate_feature.dropna(axis=1, thresh=2, inplace=True)
#validate_feature.dropna(how='all',thresh = 20,inplace=True)
validate_feature = np.array(validate_feature)
wh_inf = np.isinf(validate_feature)
validate_feature[wh_inf]=0
wh_nan = np.isnan(validate_feature)
validate_feature[wh_nan]=0

test_feature = pd.DataFrame(test_feature)
test_feature.dropna(axis=1, thresh=2, inplace=True)
#test_feature.dropna(how='all',thresh = 20,inplace=True)
test_feature = np.array(test_feature)
wh_inf = np.isinf(test_feature)
test_feature[wh_inf]=0
wh_nan = np.isnan(test_feature)
test_feature[wh_nan]=0

#only use image features
X_train = train_feature[:,:6093]
Y_train = train_feature[:,6093]
Y_train = Y_train.astype('int32')

X_validate = validate_feature[:,:6093]
Y_validate = validate_feature[:,6093]
Y_validate = Y_validate.astype('int32')

X_test = test_feature[:,:6093]
Y_test = test_feature[:,6093]
Y_test = Y_test.astype('int32')
seed = 7

np.random.seed(seed)
np.random.shuffle(X_train) 
np.random.seed(seed)
np.random.shuffle(Y_train)


models = []
models.append(('GLM', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('BY', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('BAG', BaggingClassifier()))
models.append(('NNet', MLPClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('BST', AdaBoostClassifier()))

seed = 7

#crossvalidation on trainset and select the best model on validation set, test on test set
import xlwt
sel = []
sel.append(('CHSQ', SelectKBest(chi2, k=num_fea)))
sel.append(('ANOVA', SelectKBest(f_classif, k=num_fea)))
sel.append(('TSCR', SelectKBest(t_score.t_score, k=num_fea)))
sel.append(('FSCR', SelectKBest(fisher_score.fisher_score, k=num_fea)))
sel.append(('RELF', SelectKBest(reliefF.reliefF, k=num_fea)))



book = xlwt.Workbook()
sheet = book.add_sheet('train_avg_auc')
sheet_train = book.add_sheet('train_auc')
sheet_validate = book.add_sheet('validate_auc')
sheet_test = book.add_sheet('test_auc')


from sklearn.externals import joblib
#from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
kfold = model_selection.KFold(n_splits=5, random_state=seed)
r = 0
c = 0
for name, model in models:
    for kind, selection in sel:
        pipe = make_pipeline(MinMaxScaler(), selection, model)
        cv_results = model_selection.cross_val_score(pipe, X_train, Y_train, scoring='roc_auc', cv=kfold)
        
        sheet.write(r,c,cv_results.mean())

        print("###########################################")
        msg = "%s %s %s: %f (%f)\n" % ("TRAIN_AUC", kind, name, cv_results.mean(), cv_results.std())
        print(msg)
        output.write(msg)
        pipe.fit(X_train,Y_train)
        joblib.dump(pipe,'./handpkl/Endo'+name+kind+name_str)
        
        Y_pred = pipe.predict(X_train)
        print("Accuracy: " + repr(accuracy_score(Y_train, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_train, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_train, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_train, Y_pred)))
        print("AUC: " + repr(roc_auc_score(Y_train, Y_pred)))
        print("Sensitivity: " + repr(recall_score(Y_train, Y_pred)))
        tn, fp, fn, tp = confusion_matrix(Y_train, Y_pred).ravel()
        print("Specificity: " + repr(tn / (tn + fp)))
        
        sheet_train.write(r,c,roc_auc_score(Y_train, Y_pred))

        Y_pred = pipe.predict(X_validate)
        print("Accuracy: " + repr(accuracy_score(Y_validate, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_validate, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_validate, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_validate, Y_pred)))
        print("AUC"+repr(roc_auc_score(Y_validate,Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_validate,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_validate,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_validate.write(r,c,roc_auc_score(Y_validate,Y_pred))
        
        Y_pred = pipe.predict(X_test)
        print("Accuracy: " + repr(accuracy_score(Y_test, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_test, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_test, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_test, Y_pred)))
        print("AUC: " + repr(roc_auc_score(Y_test, Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_test,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_test,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_test.write(r,c,roc_auc_score(Y_test,Y_pred))
        
        r = r + 1
    c = c + 1
    r = 0
    
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.fit_transform(X_validate)
X_test = scaler.fit_transform(X_test)

# WILCOXON SCORE FUNCTION
def takeSecond(elem):
    return elem[1]
def WLCX(data, target, n_selected_features):
    pval = []
    for num in range(len(data[1])):
        x = data[:,num]
        pval.append([num, wilcoxon(x,target)[1]])
    pval.sort(key=takeSecond)
    idx = []
    for i in range(n_selected_features):
        idx.append(pval[i][0])
    return idx

# MULTIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

# print('BEFORE')
MV_sel = []
MV_sel.append(('WLCX', WLCX(X_train, Y_train, n_selected_features=num_fea)))
print('WLCX')
for name, model in models:
    for kind, idx in MV_sel:
        # X_sel = X[:, idx[0:num_fea]]
        X_test_ = X_test[:,idx[0:num_fea]]
        X_validate_ = X_validate[:,idx[0:num_fea]]
        X_train_ = X_train[:, idx[0:num_fea]]
        # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size, random_state=seed)
        #kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train_, Y_train, cv=kfold, scoring='roc_auc')
        
        sheet.write(r,c,cv_results.mean())
        
        msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
        print(msg)
        output.write(msg)
        model.fit(X_train_, Y_train)
        joblib.dump(model,'./handpkl/Endo'+name+kind+name_str)
        
        Y_pred = model.predict(X_train_)
        print("Accuracy: " + repr(accuracy_score(Y_train, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_train, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_train, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_train, Y_pred)))
        print("AUC: " + repr(roc_auc_score(Y_train, Y_pred)))
        print("Sensitivity: " + repr(recall_score(Y_train, Y_pred)))
        tn, fp, fn, tp = confusion_matrix(Y_train, Y_pred).ravel()
        print("Specificity: " + repr(tn / (tn + fp)))
        
        sheet_train.write(r,c,roc_auc_score(Y_train, Y_pred))
        
        Y_pred = model.predict(X_validate_)    
        print("Accuracy: " + repr(accuracy_score(Y_validate, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_validate, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_validate, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_validate, Y_pred)))
        print("AUC"+repr(roc_auc_score(Y_validate,Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_validate,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_validate,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_validate.write(r,c,roc_auc_score(Y_validate,Y_pred))
        
        Y_pred = model.predict(X_test_)
        print("Accuracy: " + repr(accuracy_score(Y_test, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_test, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_test, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_test, Y_pred)))
        print("AUC"+repr(roc_auc_score(Y_test,Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_test,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_test,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_test.write(r,c,roc_auc_score(Y_test,Y_pred))
        
        r = r + 1
    c = c + 1
    r = 0


MV_sel = []
MV_sel.append(('MIM', MIM.mim(X_train, Y_train, n_selected_features=num_fea)))
print('MIM')
MV_sel.append(('MIFS', MIFS.mifs(X_train, Y_train, n_selected_features=num_fea)))
print('MIFS')
MV_sel.append(('MRMR', MRMR.mrmr(X_train, Y_train, n_selected_features=num_fea)))
print('MRMR')
MV_sel.append(('CIFE', CIFE.cife(X_train, Y_train, n_selected_features=num_fea)))
print('CIFE')
MV_sel.append(('JMI', JMI.jmi(X_train, Y_train, n_selected_features=num_fea)))
print('JMI')
MV_sel.append(('CMIM', CMIM.cmim(X_train, Y_train, n_selected_features=num_fea)))
print('CMIM')
MV_sel.append(('ICAP', ICAP.icap(X_train, Y_train, n_selected_features=num_fea)))
print('ICAP')
MV_sel.append(('DISR', DISR.disr(X_train, Y_train, n_selected_features=num_fea)))



for name, model in models:
    for kind, idx in MV_sel:
        #print(idx[0:num_fea][0])
        # X_sel = X[:, idx[0:num_fea]]
        X_test_ = X_test[:,idx[0:num_fea]]
        X_validate_ = X_validate[:,idx[0:num_fea]]
        X_train_ = X_train[:, idx[0:num_fea]]
        # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size, random_state=seed)
        #kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train_, Y_train, cv=kfold, scoring='roc_auc')
        
        sheet.write(r,c,cv_results.mean())
        
        msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
        print(msg)
        output.write(msg)
        model.fit(X_train_, Y_train)
        joblib.dump(model,'./handpkl/Endo'+name+kind+name_str)
        
        Y_pred = model.predict(X_train_)
        print("Accuracy: " + repr(accuracy_score(Y_train, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_train, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_train, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_train, Y_pred)))
        print("AUC: " + repr(roc_auc_score(Y_train, Y_pred)))
        print("Sensitivity: " + repr(recall_score(Y_train, Y_pred)))
        tn, fp, fn, tp = confusion_matrix(Y_train, Y_pred).ravel()
        print("Specificity: " + repr(tn / (tn + fp)))
        
        sheet_train.write(r,c,roc_auc_score(Y_train, Y_pred))
        
        Y_pred = model.predict(X_validate_)    
        print("Accuracy: " + repr(accuracy_score(Y_validate, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_validate, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_validate, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_validate, Y_pred)))
        print("AUC"+repr(roc_auc_score(Y_validate,Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_validate,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_validate,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_validate.write(r,c,roc_auc_score(Y_validate,Y_pred))
        
        Y_pred = model.predict(X_test_)
        print("Accuracy: " + repr(accuracy_score(Y_test, Y_pred)))
        print("Average Precision Score: " + repr(average_precision_score(Y_test, Y_pred)))
        print("Kappa: " + repr(cohen_kappa_score(Y_test, Y_pred)))
        print("Hamming Loss: " + repr(hamming_loss(Y_test, Y_pred)))
        print("AUC"+repr(roc_auc_score(Y_test,Y_pred)))
        print("Sensitivity" + repr(recall_score(Y_test,Y_pred)))
        tn,fp,fn,tp = confusion_matrix(Y_test,Y_pred).ravel()
        print("Specificity" + repr(tn/(tn+fp)))
        
        sheet_test.write(r,c,roc_auc_score(Y_test,Y_pred))
        
        r = r + 1
    c = c + 1
    r = 0
book.save(file)
output.close()
