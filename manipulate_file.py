import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from collections import defaultdict
def vali(x):
    if x is None:
        ss={"status":"false","msg":"uploaded file has no column headers...upload valid file to build the model"}
        result=json.dumps(ss)
        return result
    else:
        return "continue"
def parse(path,ex,mid,email):
    print(path)
    if ex in ['.xlsx']:
        df=pd.read_excel(path,encoding='latin')
    else:
       json_res =parse_csv(path,ex,email,mid)
       return json_res
    cols=df.shape
    check_cols=cols[1]
    check_rows=cols[0]
    print("mancols")
    if check_cols>1:
        if check_rows>=30:           
            ex_path=Path(path)
            dy_folder=str(ex_path.parts[6])
    #ss = re.findall('uid:(.+)',dy_folder)
            x_list=list(df.columns)
            col_type={}
            for i in x_list:
                col_type[i]=str(df[i].dtypes)
        #print(t)
            res=vali(x_list)
            if res != 'continue':
                return res
            else:      
                y_list=list(df.columns)
                type(y_list)
                algo_classifier=[]
                algo_regressor=[] 
                algo_class=defaultdict(dict)
                dic={"name":"max_depth","type":"range","minval":2,"pval":10}
                dic1={"name":"max_features","type":"list","pval":['auto','sqrt','log2']}
                dic2={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic3={"name":"criterion","type":"list","pval":['gini','entropy']}
                dic4={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic5={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                algo_class['DecisionTreeClassifier']={"hyperparameters":[dic,dic1,dic2,dic3,dic4]}
                dic1_rf={"name":"n_estimators","type":"range","minval":2,"pval":1000}
                dic2_rf={"name":"max_features","type":"list","pval":['auto','sqrt','log2']}
                dic4_rf={"name":"min_samples_leaf","type":"range","minval":2,"pval":50}
                dic5_rf={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic6_rf={"name":"max_depth","type":"range","minval":2,"pval":50}
                dic3_rf={"name":"criterion","type":"list","pval":['gini','entropy']}
                algo_class['RandomForestClassifier']={"hyperparameters":[dic1_rf,dic2_rf,dic3_rf,dic4_rf,dic5_rf,dic6_rf]}
                dic1_svm={"name":"C","type":"range","minval":1,"pval":1000}
                dic2_svm={"name":"gamma","type":"range","minval":1,"pval":10}
                algo_class['svm-rbf']={"hyperparameters":[dic1_svm,dic2_svm]}
                dic1_svml={"name":"C","type":"range","minval":1,"pval":1000}
                dic2_svml={"name":"class_weight","type":"list","pval":[None,'balanced']}
                dic3_svml={"name":"multi_class","type":"list","pval":['ovr','crammer_singer']}
                algo_class['svm-linear']={"hyperparameters":[dic1_svml,dic2_svml,dic3_svml]}
                dic1_kn={"name":"n_neighbors","type":"range","minval":2,"pval":30}
                #dic2_kn={"name":"class_weight","type":"list","pval":['None','balanced']}
                dic3_kn={"name":"algorithm","type":"list","pval":['auto','ball_tree','kd_tree','brute']}
                dic4_kn={"name":"p","type":"list","pval":[1,2]}
                algo_class['KNeighborsClassifier']={'hyperparameters':[dic1_kn,dic3_kn,dic4_kn]}
                algo_class['GaussianNBclassifier']={'hyperparameters':[]}
                algo_classifier.append('DecisionTreeClassifier')
                algo_classifier.append('RandomForestClassifier')
                algo_classifier.append('svm-rbf')
                algo_classifier.append('svm-linear')
                algo_classifier.append('KNeighborsClassifier')
                algo_classifier.append('GaussianNBclassifier')
                algo_regres=defaultdict(dict)
                dic1_lir={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_lir={"name":"n_jobs","type":"list","pval":[1,2,3,4,-1]}
                dic3_lir={"name":"fit_intercept","type":"list","pval":["True","False"]}
                algo_regres["LinearRegressor"]={'hyperparameters':[dic1_lir,dic2_lir,dic3_lir]}
                dic1_rr={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_rr={"name":"solver","type":"list","pval":["auto","svd","cholesky","lsqr","sparse_cg","sag","saga"]}
                dic3_rr={"name":"alpha","type":"range","minval":0.1,"pval":1}
                algo_regres['RidgeRegressor']={'hyperparameters':[dic1_rr,dic2_rr,dic3_rr]}
                dic0_dtr={"name":"max_depth","type":"range","minval":2,"pval":10}
                dic1_dtr={"name":"max_features","type":"range","minval":2,"pval":10}
                dic2_dtr={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic4_dtr={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic5_dtr={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic3_dtr={"name":"criterion","type":"list","pval":["friedman_mse","mae"]}
                algo_regres['DecisiontreeRegressor']={'hyperparameters':[dic1_dtr,dic2_dtr,dic3_dtr,dic4_dtr,dic5_dtr,dic0_dtr]}
                dic1_rfr={"name":"n_estimators","type":"range","minval":2,"pval":1000}
                dic2_rfr={"name":"max_features","type":"list","pval":['all','auto','sqrt','log2']}
                dic4_rfr={"name":"min_samples_leaf","type":"range","minval":2,"pval":50}
                dic5_rfr={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic6_rfr={"name":"max_depth","type":"range","minval":2,"pval":50}
                dic3_rfr={"name":"criterion","type":"list","pval":["mse","mae"]}
                algo_regres['RandomforestRegressor']={'hyperparameters':[dic1_rfr,dic2_rfr,dic3_rfr,dic4_rfr,dic5_rfr,dic6_rfr]}
                dic1_lor={"name":"penalty","type":"list","pval":["l1","l2"]}
                dic2_lor={"name":"C","type":"range","minval":1,"pval":10}
                dic4_lor={"name":"multi_class","type":"list","pval":["ovr","multinomial"]}
                dic3_lor={"name":"solver","type":"list","pval":["newton-cg","lbfgs","liblinear","sag","saga"]}
                algo_regres['LogisticRegressor']={'hyperparameters':[dic1_lor,dic2_lor,dic3_lor,dic4_lor]}
                dic1_lar={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_lar={"name":"positive","type":"list","pval":["True","False"]}
                dic3_lar={"name":"alpha","type":"range","minval":0.1,"pval":1}
                algo_regres['LassoRegressor']={'hyperparameters':[dic1_lar,dic2_lar,dic3_lar]}
                algo_classifier.append('DecisionTreeClassifier')
                algo_classifier.append('RandomForestClassifier')
                algo_classifier.append('svm-rbf')
                algo_classifier.append('svm-linear')
                algo_classifier.append('KNeighborsClassifier')
                algo_classifier.append('GaussianNBclassifier')
                algo_regressor.append('LinearRegressor')
                algo_regressor.append('RidgeRegressor')
                algo_regressor.append('DecisiontreeRegressor')
                algo_regressor.append('RandomforestRegressor')
                algo_regressor.append('LogisticRegressor')
                algo_regressor.append('LassoRegressor')
                tt_split=[]
                tt_split.append("80% train and 20% test")
                tt_split.append("70% train and 30% test")
                tt_split.append("50% train and 50% test")
                ml_type=[]
                ml_type.append("Auto-AI")
                ml_type.append("let AI predict")
                obj = {
                "status":"true",
                "msg":"successfully upload completed",
                "modelId":mid,
                "email":email,
                "filePath":path,
                "fileExtension":ex,
                "datatypes":col_type,
                "prob_type":ml_type,"tt_split":tt_split,
                "algo":{"classification_hyper":algo_class,"regression_hyper":algo_regres}}
                json_data = json.dumps(obj)
                print(col_type)
                print(type(obj))
                return json_data
        else:
            json_fail2={"status":"false","msg":"your dataset should have atleast 30 records to build the model"}
            res3=json.dumps(json_fail2)
            return res3
    else:
        json_fail2={"status":"false","msg":"your dataset should have atleast 2 columns to build the model"}
        res3=json.dumps(json_fail2)
        return res3

def parse_csv(path,ex,email,mid):
    df=pd.read_csv(path,encoding='latin')
    cols=df.shape
    check_cols=cols[1]
    check_rows=cols[0]
    print("man col")
    if check_cols>1:
        if check_rows>=30:
            x_list=list(df.columns)
            res=vali(x_list)
            if res !='continue':
                return res
            else:
                col_type={}
                for i in x_list:
                    col_type[i]=str(df[i].dtypes)
                y_list=list(df.columns)
                ex_path=Path(path)
                dy_folder=str(ex_path.parts[6])
        #ss = re.findall('uid:(.+)',dy_folder)
                algo_classifier=[]
                algo_regressor=[]
                algo_class=defaultdict(dict)
                dic={"name":"max_depth","type":"range","minval":2,"pval":10}
                dic1={"name":"max_features","type":"list","pval":['auto','sqrt','log2']}
                dic2={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic3={"name":"criterion","type":"list","pval":['gini','entropy']}
                dic4={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic5={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                algo_class['DecisionTreeClassifier']={"hyperparameters":[dic,dic1,dic2,dic3,dic4]}
                dic1_rf={"name":"n_estimators","type":"range","minval":10,"pval":1000}
                dic2_rf={"name":"max_features","type":"list","pval":['auto','sqrt','log2']}
                dic4_rf={"name":"min_samples_leaf","type":"range","minval":2,"pval":50}
                dic5_rf={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic6_rf={"name":"max_depth","type":"range","minval":2,"pval":50}
                dic3_rf={"name":"criterion","type":"list","pval":['gini','entropy']}
                algo_class['RandomForestClassifier']={"hyperparameters":[dic1_rf,dic2_rf,dic3_rf,dic4_rf,dic5_rf,dic6_rf]}
                dic1_svm={"name":"C","type":"range","minval":2,"pval":1000}
                dic2_svm={"name":"gamma","type":"range","minval":2,"pval":10}
                algo_class['svm-rbf']={"hyperparameters":[dic1_svm,dic2_svm]}
                dic1_svml={"name":"C","type":"range","minval":2,"pval":1000}
                dic2_svml={"name":"class_weight","type":"list","pval":[None,'balanced']}
                dic3_svml={"name":"multi_class","type":"list","pval":['ovr','crammer_singer']}
                algo_class['svm-linear']={"hyperparameters":[dic1_svml,dic2_svml,dic3_svml]}
                dic1_kn={"name":"n_neighbors","type":"range","minval":2,"pval":30}
                dic2_kn={"name":"class_weight","type":"list","pval":[None,'balanced']}
                dic3_kn={"name":"algorithm","type":"list","pval":['auto','ball_tree','kd_tree','brute']}
                dic4_kn={"name":"p","type":"list","pval":[1,2]}
                algo_class['KNeighborsClassifier']={'hyperparameters':[dic1_kn,dic3_kn,dic4_kn]}
                algo_class['GaussianNBclassifier']={'hyperparameters':[]}
                algo_classifier.append('DecisionTreeClassifier')
                algo_classifier.append('RandomForestClassifier')
                algo_classifier.append('svm-rbf')
                algo_classifier.append('svm-linear')
                algo_classifier.append('KNeighborsClassifier')
                algo_classifier.append('GaussianNBclassifier')
                algo_regres=defaultdict(dict)
                dic1_lir={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_lir={"name":"n_jobs","type":"list","pval":[1,2,3,4,-1]}
                dic3_lir={"name":"fit_intercept","type":"list","pval":["True","False"]}
                algo_regres["LinearRegressor"]={'hyperparameters':[dic1_lir,dic2_lir,dic3_lir]}
                dic1_rr={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_rr={"name":"solver","type":"list","pval":["auto","svd","cholesky","lsqr","sparse_cg","sag","saga"]}
                dic3_rr={"name":"alpha","type":"range","minval":0.1,"pval":1}
                algo_regres['RidgeRegressor']={'hyperparameters':[dic1_rr,dic2_rr,dic3_rr]}
                dic0_dtr={"name":"max_depth","type":"range","minval":2,"pval":10}
                dic1_dtr={"name":"max_features","type":"range","minval":2,"pval":10}
                dic2_dtr={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic4_dtr={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic5_dtr={"name":"min_samples_leaf","type":"range","minval":2,"pval":10}
                dic3_dtr={"name":"criterion","type":"list","pval":["friedman_mse","mae"]}
                algo_regres['DecisiontreeRegressor']={'hyperparameters':[dic1_dtr,dic2_dtr,dic3_dtr,dic4_dtr,dic5_dtr,dic0_dtr]}
                dic1_rfr={"name":"n_estimators","type":"range","minval":2,"pval":1000}
                dic2_rfr={"name":"max_features","type":"list","pval":['all','auto','sqrt','log2']}
                dic4_rfr={"name":"min_samples_leaf","type":"range","minval":2,"pval":50}
                dic5_rfr={"name":"min_samples_split","type":"range","minval":2,"pval":50}
                dic6_rfr={"name":"max_depth","type":"range","minval":2,"pval":50}
                dic3_rfr={"name":"criterion","type":"list","pval":["mse","mae"]}
                algo_regres['RandomforestRegressor']={'hyperparameters':[dic1_rfr,dic2_rfr,dic3_rfr,dic4_rfr,dic5_rfr,dic6_rfr]}
                dic1_lor={"name":"penalty","type":"list","pval":["l1","l2"]}
                dic2_lor={"name":"C","type":"range","minval":2,"pval":10}
                dic4_lor={"name":"multi_class","type":"list","pval":["ovr","multinomial"]}
                dic3_lor={"name":"solver","type":"list","pval":["newton-cg","lbfgs","liblinear","sag","saga"]}
                algo_regres['LogisticRegressor']={'hyperparameters':[dic1_lor,dic2_lor,dic3_lor,dic4_lor]}
                dic1_lar={"name":"normalize","type":"list","pval":["True","False"]}
                dic2_lar={"name":"positive","type":"list","pval":["True","False"]}
                dic3_lar={"name":"alpha","type":"range","minval":0.1,"pval":1}
                algo_regres['LassoRegressor']={'hyperparameters':[dic1_lar,dic2_lar,dic3_lar]}
                algo_regressor.append('LinearRegressor')
                algo_regressor.append('RidgeRegressor')
                algo_regressor.append('DecisiontreeRegressor')
                algo_regressor.append('RandomforestRegressor')
                algo_regressor.append('LogisticRegressor')
                algo_regressor.append('LassoRegressor')
                tt_split=[]
                tt_split.append("80% train and 20% test")   
                tt_split.append("70% train and 30% test")
                tt_split.append("50% train and 50% test")
                ml_type=[]
                ml_type.append("Auto-AI")
                ml_type.append("Let AI Predict")
                print(col_type)
                obj = {
                   "status":"true",
                   "msg":"successfully upload completed",
                   "modelId":mid,
                   "email":email,
                   "filePath":path,
                   "fileExtension":ex,
                   "datatypes":col_type,
                   "prob_type":ml_type,"tt_split":tt_split,
                   "algo":{"classification_hyper":algo_class,"regression_hyper":algo_regres}
                   } 
                json_data = json.dumps(obj)
                print(json_data)
                return json_data
        else:
            json_fail2={"status":"false","msg":"your dataset should have atleast 30 records to build the model"}
            res3=json.dumps(json_fail2)
            return res3
    else:
        json_fail2={"status":"false","msg":"your dataset should have atleast 2 columns to build the model"}
        res3=json.dumps(json_fail2)
        return res3
   
