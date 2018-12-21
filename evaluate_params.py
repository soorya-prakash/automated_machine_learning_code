from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import pickle
import json
import re
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
from collections import OrderedDict
dummy=pd.DataFrame()
X_train=pd.DataFrame()
y_train=pd.DataFrame()
X_test=pd.DataFrame()
y_test=pd.DataFrame()
label_encoder = LabelEncoder()
label_encoder1=LabelEncoder()
label_encoderx=LabelEncoder()
filepath1=[]
feature_count=""
sel_x=()
test_df=pd.DataFrame()
target_df=pd.DataFrame()
comman_list=[]
tt=[]
RANDOM_STATE=10
def val(n):
    if n == 0:
        print("entered into if")
        json_fail={"status":"false","msg":"0 rows after preprocessing ,choose target and x features carefully"}
        result_json=json.dumps(json_fail)
        print(result_json)
        return result_json
    else:
        return 'continue'
def hint(df,x,y):
    record=list(df.iloc[0])
    ex_dict=OrderedDict()
    for i in range(len(record)):
        ex_dict[x[i]]=str(record[i])
    print(ex_dict)
    ex_dict.pop(y, None)
    print(ex_dict)
    return ex_dict    
def preprocess_dropna2(filepath,ex,x,y,algo,tt_split,mid,algo_params,email,label_encoder_file_path):     
    global dummy,filepath1,feature_count,sel_x,target_df,comman_list,tt
    tt=tt_split
    print('came in',label_encoder_file_path)
    sel_x=x
    #print(sel_x)
    feature_count=int(len(x))
    #print(feature_count)
    filepath1=filepath
    if ex in ['.csv']:
        try:
            df_preprocess=pd.read_csv(filepath,encoding='raw_unicode_escape')
        except UnicodeDecodeError:
            failure={"status":"false","msg":"uploaded file cannot be used for modelling pls upload proper file"}
            success_json=json.dumps(failure)
            return success_json 

        list_x=x
       # print(list_x)
        list_y=str(y)
        target_df=pd.DataFrame({'target':df_preprocess[list_y]})
        comman_list=x
        comman_list.append(list_y)
        print(comman_list)
        required_df=df_preprocess[comman_list]
        required_df=required_df.dropna(axis=0)
        shape=required_df.shape
        print(shape)
        zero=shape[0]
        print(zero)
        res=val(zero)
        if res!='continue':
            return res

       # print(res)
        #return res
        else:
            ex_dict=hint(required_df,x,y)
            dummy=dummy.append(required_df)
            visual()
            print('yet to encode')
            encoding(required_df,filepath,label_encoder_file_path)
        #normalize()
        #print(list_x)
        #print(list_y)
            print('encoding done')
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            print(x1)
            data_spilt(x1,y1)
            result_json=model(algo,ex_dict,mid,algo_params,email)
            return result_json
    else: 
        df_preprocess=pd.read_excel(filepath,encoding='latin')
        list_x=x
        list_y=y
        comman_list=list_x
        comman_list.append(list_y)
        print(comman_list)
        required_df=df_preprocess[comman_list]
        required_df=required_df.dropna()
        print(required_df.shape)
        shape=required_df.shape
        zero=shape[0]
        res=val(zero)
        if res!='continue':
            return res
        else:
            ex_dict=hint(required_df,x,y)
            dummy=dummy.append(required_df)
            visual()
            encoding(required_df,filepath,label_encoder_file_path)
            #normalize()
            print(target_df)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            data_spilt(x1,y1)
            result_json=model(algo,ex_dict,mid,algo_params,email)
            return result_json
def visual():
    print(dummy)
def encoder_using_path(le,df_x):
    for i in df_x.select_dtypes(['object']):
        if i in le:
            temp =le.get(i)
            # my_dict2= {y:x for x,y in temp.items()}
            # label[i]=dict(my_dict2)
            df_x[i] = df_x[i].map(temp) 
            print("encoded" ,i)
    return df_x
def encoding(required_df,filepath,label_encoder_file_path):
    global dummy
    print(label_encoder_file_path)
    with open(label_encoder_file_path, 'rb') as file:  
        le=pickle.load(file)
    req_df=encoder_using_path(le,required_df)
    dummy=req_df
def normalize():
    global dummy
    scaler = StandardScaler()
    scaled_features=scaler.fit_transform(dummy)
    scaled_features_df = pd.DataFrame(scaled_features, index=dummy.index, columns=dummy.columns)
    dummy =scaled_features_df
    visual()
def data_spilt(x,y):
    print("test split",x)
    global X_train,X_test,y_train,y_test,tt
    if "80% train and 20% test" in tt:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    elif "70% train and 30% test" in tt:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    elif "50% train and 50% test" in tt:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
def algo_select(li):
    global X_train,y_train,X_test,y_test
    # for i in li:
    #     if "DecisionTreeClassifier" in i:
    #         para=[]
    #         para_value=[]
    #         for j in li:
    #             par=li[j]
    #             for p in par:
    #                 para_value.append(par.get(p))
    #             clf = tree.DecisionTreeClassifier(p=v)
    #             clf = clf.fit(X_train,y_train)
    #             pre=clf.predict(X_test)
    #             acc=accuracy_score(y_test,pre)
    #             print("training done")
    #             return acc*100
    for u in li:
        algo_tune=u
        if "DecisionTreeClassifier" in algo_tune:
            var=[]
            keys=[]
            for key,value in li[u].items():
                if value !="":
                    var.append(key)
                    keys.append(value)
            print(var,keys)
            if "max_depth" in var:
                indi=var.index('max_depth')
                max_depths=int(keys[indi])
            else:
                max_depths=None
            if "max_features" in var:
                indi1=var.index('max_features')
                max_featuress=keys[indi1]
            else:
                max_featuress=None 
            if "min_samples_leaf" in var:
                indi2=var.index('min_samples_leaf')
                min_samples_leafs=int(keys[indi2])
            else:
                min_samples_leafs=1
            if "criterion" in var:
                indi3=var.index('criterion')
                criterions=keys[indi3]
            else:
                criterions='gini'
            if "min_samples_split" in var:
                indi4=var.index('min_samples_split')
                min_samples_splits=int(keys[indi4])
            else:
                min_samples_splits=2
            clf = tree.DecisionTreeClassifier(max_depth=max_depths,max_features=max_featuress,min_samples_leaf=min_samples_leafs,criterion=criterions,min_samples_split=min_samples_splits,random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            pre=clf.predict(X_test)
            acc=accuracy_score(y_test,pre)
            print("training done",acc)
            return acc
        if "RandomForestClassifier" in algo_tune:
            var1=[]
            keys1=[]
            for key,value in li[u].items():
                if value !="":
                    var1.append(key)
                    keys1.append(value)
            if "max_depth" in var1:
                indi8=var1.index('max_depth')
                max_depths=keys1[indi8]
            else:
                max_depths=None
            if "n_estimators" in var1:
                indi5=var1.index('n_estimators')
                n_estimatorss=int(keys1[indi5])
            else:
                n_estimatorss=10
            if "min_samples_leaf" in var1:
                indi6=var1.index('min_samples_leaf')
                min_samples_leafs=int(keys1[indi6])
            else:
                min_samples_leafs=1
            if "min_samples_split" in var1:
                indi7=var1.index('min_samples_split')
                min_samples_splits=int(keys1[indi7])
            else:
                min_samples_splits=2
            if "criterion" in var1:
                indi9=var1.index('criterion')
                criterions=keys1[indi9]
            else:
                criterions='gini'
            if "max_features" in var1:
                indi10=var1.index('max_features')
                max_featuress=keys1[indi10]
            else:
                max_featuress=None 
            clf1=RandomForestClassifier(n_estimators=n_estimatorss,max_depth=max_depths,min_samples_split=min_samples_splits,min_samples_leaf=min_samples_leafs,max_features=max_featuress,criterion=criterions,random_state=RANDOM_STATE)
            clf1 = clf1.fit(X_train,y_train)
            pre=clf1.predict(X_test)
            acc=accuracy_score(y_test,pre)
            print(acc)
            print("training done")
            return acc
     
        if "svm-rbf" in algo_tune:   
            var3=[]
            keys3=[]
            for key,value in li[u].items():
                if value !="":
                    var3.append(key)
                    keys3.append(value)
            if "C" in var3:
                indi11=var3.index('C')
                c=int(keys3[indi11])
            else:
                c=1.0
            if "gamma" in var3:
                indi12=var3.index('gamma')
                gammas=keys3[indi12]
            else:
                gammas='auto'
            clf1=svm.SVC(kernel='rbf', gamma=gammas, C=c,random_state=RANDOM_STATE)
            clf1 = clf1.fit(X_train,y_train)
            pre=clf1.predict(X_test)
            acc=accuracy_score(y_test,pre)
            print(acc)
            print("training done")
            return acc
        if "svm-linear" in algo_tune:
            var4=[]
            keys4=[]
            for key,value in li[u].items():
                if value !="":
                    var4.append(key)
                    keys4.append(value)
            if "C" in var4:
                indi13=var4.index('C')
                c=int(keys4[indi13])
            else:
                c=1.0
            if 'class_weight' in var4:
                indi14=var4.index('class_weight')
                class_weights=keys4[indi14]
            else:
                class_weights=None
            if "multi_class" in var4:
                indi15=var4.index('multi_class') 
                multi_classs=keys4[indi15]
            else:
                multi_classs='ovr'
            clf1=svm.LinearSVC(C=c,class_weight=class_weights,multi_class=multi_classs,random_state=RANDOM_STATE)
            clf1 = clf1.fit(X_train,y_train)
            pre=clf1.predict(X_test)
            acc=accuracy_score(y_test,pre)
            print(acc)
            print("training done")
            return acc   
        if "KNeighborsClassifier" in algo_tune:
            var5=[]
            keys5=[]
            for key,value in li[u].items():
                if value !="":
                    var5.append(key)
                    keys5.append(value)
            if 'n_neighbors' in var5:
                indi16=var5.index('n_neighbors')
                n_neighborss=int(keys5[indi16])
            else:
                n_neighborss=5
           # if 'class_weight' in var5:
           #     indi17=var5.index('class_weight')
           #     class_weights=keys5[indi17]
           # else:
           #     class_weights=None
            if 'algorithm' in var5:
                indi18=var5.index("algorithm")
                algorithms=keys5[indi18]
            else:
                algorithms='ovr'
            if 'p' in var5:
                indi19=var5.index('p')
                ps=int(keys5[indi19])
            else:
                ps=2
            clf1=KNeighborsClassifier(n_neighbors=n_neighborss,algorithm=algorithms,p=ps)
            clf1 = clf1.fit(X_train,y_train)
            pre=clf1.predict(X_test)
            acc=accuracy_score(y_test,pre)
            print(acc)
            print("training done")
            return acc  
def model(algo,ex_dict,mid,algo_params,email):
    global X_train,X_test,y_train,y_test,filepath1,feature_count,sel_x,dummy,tt
    print(algo_params)
    li={}
    acc={}
    for i in algo_params:
        li[i]=algo_params[i]
        if "classification" in algo:
            acc[i]=algo_select(li)
            li.clear()
        else:
            acc[i]=algo_selectr(li)
            li.clear()
    print(acc)
    ss={"modelId":mid,"algo":algo,"accuracy":acc,"email":email,"status":"True"}
    success_json=json.dumps(ss)
    return success_json
 


   

