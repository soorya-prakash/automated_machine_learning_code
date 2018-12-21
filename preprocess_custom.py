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
RANDOM_STATE =0
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
def preprocess_dropna(filepath,ex,x,y,algo,tt_split,mid,email,yobj,type,deploy,algo_params,valu,label_encoder_path):     
    global dummy,filepath1,feature_count,sel_x,target_df,comman_list,tt
    if "auto" in type:
        tt.append("80% train and 20% test")
    elif "custom" in type:
        tt.append(tt_split)
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
            print("see")
            visual()
            encoding(required_df,label_encoder_path)
        #normalize()
        #print(list_x)
        #print(list_y)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            print(x1)
            data_spilt(x1,y1)
            result_json=model(algo,email,yobj,ex_dict,mid,type,filepath,deploy,algo_params,valu,label_encoder_path)
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
            encoding(required_df,label_encoder_path)
            #normalize()
            print(target_df)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            data_spilt(x1,y1)
            result_json=model(algo,email,yobj,ex_dict,mid,type,filepath,deploy,algo_params,valu,label_encoder_path)
            return result_json
def visual():
    print(dummy)
def encoder_using_path(le,df_x):
    for i in df_x.select_dtypes(['object']):
        if i in le.keys():
            temp =le.get(i)
            # my_dict2= {y:x for x,y in temp.items()}
            # label[i]=dict(my_dict2)
            df_x[i] = df_x[i].map(temp) 
            print("encoded" ,i)
    return df_x
def encoding(required_df,label_encoder_path):
    global dummy
    with open(label_encoder_path, 'rb') as file:  
        le=pickle.load(file)
    req_df=encoder_using_path(le,required_df)
#     global label_encoder
#    # required_df=required_df.iloc[:,0:-1]
#     for c in required_df.select_dtypes(['object']):
#         print ("Encoding column " + c)
#         required_df[c] = label_encoder.fit_transform(required_df[c].astype(str))
#         np.save('classes.npy',label_encoder.classes_)  
#     req=required_df
#     req=required_df.iloc[:,0:-1]
   # for c in required_df.select_dtypes(['object']):
     #   print ("Encoding column " + c)
    #    required_df[c] = label_encoderx.fit_transform(required_df[c].astype(str))
   #     np.save('classes.npy',label_encoderx.classes_)
  #  for i in target_df.columns:
 #       target_df[i]=label_encoder1.fit(target_df[i].astype(str))
#        np.save('classes.npy', label_encoder1.classes_)
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
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=RANDOM_STATE)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    elif "70% train and 30% test" in tt:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=RANDOM_STATE)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    elif "50% train and 50% test" in tt:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50,random_state=RANDOM_STATE)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
def model(algo,email,obj,ex_dict,mid,type,filepath,deploy,algo_params,valu,label_encoder_path):
    global X_train,X_test,y_train,y_test,filepath1,feature_count,sel_x,dummy,tt
    print(X_test)
    if "80% train and 20% test" in tt:
        train_split=80
        test_split=100-train_split
    elif "70% train and 30% test" in tt:
        train_split=70
        test_split=100-train_split
    elif "50% train and 50% test" in tt:
        train_split=50
        test_split=100-train_split
    #ex_path=Path(filepath1)
    #dy_folder=str(ex_path.parts[6])
    #sss = re.findall('uid:(.+)',dy_folder)
    #clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(X_train,y_train)
    #print("training done")
    print(algo)
    print(algo_params)
    clf=algo_select(algo,algo_params)
    print(clf)
    ex_path=Path(filepath1)
    fp1=str(ex_path.parts[6])
    name=str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if "int64" in obj:
        if valu==2 or valu==3 :
            pkl_filename =str("pickle_model(classification){}.pkl".format(name))
        else:
            pkl_filename =str("pickle_model(regression){}.pkl".format(name))
    elif "float64" in obj:
        pkl_filename =str("pickle_model(regression){}.pkl".format(name)) 
    else:
        pkl_filename =str("pickle_model(classification){}.pkl".format(name)) 
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(clf, file)
    
    with open(pkl_filename, 'rb') as file:  
        pickle_model = pickle.load(file)
    Ypredict = pickle_model.predict(X_test)
    score = pickle_model.score(X_test,y_test)  
    Testscore="{0:.2f} %".format(100 * score) 
    print(Testscore)
    print(Ypredict)
    print(y_test)
    # if "int64" in obj:
    #     if valu==2 or valu==3 : 
    #         res_inverse=inverse(Ypredict,y_test)
    #     else:
    #         res_inverse=inve(Ypredict,y_test)
    # elif "float64" in obj:
    #     res_inverse=inve(Ypredict,y_test)
    # else:
    #     res_inverse=inverse(Ypredict,y_test)
    # ss={"status":"True","deploy":deploy,"prediction_type":type,"model_file_path":filepath,"modelId":mid,"download":"/home/ubuntu/copy/master_api/root_files/{}/{}".format(fp1,res_inverse),"pklfile":pkl_filename,"result":res_inverse,"accuracy":Testscore,"msg":"successfully model created","train_split":train_split,"test_split":test_split,"algo":algo,"email":email,"hint":ex_dict}
    ss={"status":"True","deploy":deploy,"prediction_type":type,"label_encoder_file_path":label_encoder_path,"model_file_path":filepath,"modelId":mid,"pklfile":pkl_filename,"accuracy":Testscore,"msg":"successfully model created","train_split":train_split,"test_split":test_split,"algo":algo,"email":email,"hint":ex_dict}
    success_json=json.dumps(ss)
    print('sent success')
    return success_json
# def inve(predited,actual):
#     name=str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#     pred_df=pd.DataFrame(predited)
#     correect_df=pd.DataFrame(actual)
#     pred_df.reset_index(drop=True, inplace=True)
#     correect_df.reset_index(drop=True, inplace=True)
#     final_df=pd.concat([correect_df,pred_df],axis=1)
#     final_df.columns=['ACTUAL VALUES','PREDICTED VALUES']
#     print(final_df)
#     final_df.to_csv('result{}.csv'.format(name))
#     return "result{}.csv".format(name)
# def inverse(predicted,actual):
#     global label_encoder
#     name=str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#     pred=label_encoder.inverse_transform(predicted)
#     correct=label_encoder.inverse_transform(actual)
#     pred_df=pd.DataFrame(pred)
#     correect_df=pd.DataFrame(correct)
#     final_df=pd.concat([correect_df,pred_df],axis=1)
#     final_df.columns=['ACTUAL VALUES','PREDICTED VALUES']
#     print(final_df)
#     final_df.to_csv('result{}.csv'.format(name))
#     return "result{}.csv".format(name)
def transform_input_df(input_df,lable_encoder):
    print("before inverse transforming")
    print(input_df)
    with open(lable_encoder,'rb') as file:
        le=pickle.load(file)
    length_encoded_columns=len(list(input_df.columns))
    if length_encoded_columns != 0 and len(input_df.index) != 0:
        for i in input_df.select_dtypes(['object']):
            if i in le.keys():
                temp =le.get(i)
                my_dict2= {x:y for x,y in temp.items()}
                print(i,":",my_dict2)
                input_df[i] = input_df[i].map(my_dict2)
                print("encoded",i)
            else:
                print('column is not encoded',i)
        print("after inverse transforming")
        print(input_df)
        return input_df
    else:
        print("dataframe is empty")
def inverse_transform_y(key,Ypredict,le):
    if key in le.keys:
        temp=le.get(key)
        my_dict2= {y:x for x,y in temp.items()}
        for res in my_dict2:
            if res == Ypredict:
                result=my_dict2.get(res)
                return result
    else:
        print("y is not in label_encoder")
def prediction(pkl,record,col,mid,email,label_encoder_path,y):
    global label_encoder1,dummy,test_df,label_encoder,X_test
    row=pd.Series(record,col)
    print(row)
    test_df=pd.DataFrame()
    test_df=test_df.append([row],ignore_index=True)
    #encoding(dummy)
    #dict={col[0]:[record[0]],col[1]:[record[1]],col[2]:[record[2]],col[3]:[record[3]]}
    #test_record=pd.DataFrame(dict)
    with open(label_encoder_path, 'rb') as file:  
        le=pickle.load(file)
    print(test_df,le)
    encoded_test_df=transform_input_df(test_df,le)

    #label_encoder.classes_=np.load('classes1.npy')
    # for c in test_df.select_dtypes(['object']):
    #     print ("Encoding column " + c)
    #     test_df[c] = label_encoder.fit_transform(test_df[c].astype(str))    
#test_df=label_encoder.fit_transform(test_df) 
    #encoded_test=encoding_test(test_record,col)
    print(encoded_test_df) 
    #ex_path=Path(fp)
    #fp1=str(ex_path.parts[5])
    pkl_file="{}".format(pkl)
    with open(pkl_file,'rb') as file:
        pickle_model=pickle.load(file)
    Ypredict = pickle_model.predict(encoded_test_df)
    s=str(pkl_file)
    ss = re.findall('model(.+)2018', s)
    print(ss)
    if "(classification)" in ss:
        key=y
        predicted_result=inverse_transform_y(key,Ypredict,le)
        print(predicted_result)
        predicted=str(predicted_result)
        ss={"status":"True","result":predicted,"modelId":mid,"email":email}
        json_result=json.dumps(ss)
        return str(json_result)
    else:
        result=str(Ypredict)
        ss={"status":"True","result":result,"modelId":mid,"email":email}
        json_result=json.dumps(ss)
        return str(json_result)
def encoding_test(test_df,col):
    global label_encoder
    print(test_df)
    #test_df.columns=[col]
    for c in test_df.select_dtypes(['object']):
        print ("Encoding column " + c)
        test_df[c] = label_encoder.fit_transform(test_df[c].astype(str))
    print(test_df)
    return test_df
def algo_select(algo1,algo_params):
    global X_train,y_train,RANDOM_STATE
    if 'DecisionTreeClassifier' in algo1:
        if not algo_params:
            clf = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            print("training done")
            return clf 
        else:
            for u in algo_params:
                algo_tune=u
            if "DecisionTreeClassifier" in algo_tune:
                var=[]
                keys=[]
                for key,value in algo_params[u].items():
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
                print("training done")
                return clf
             
    elif 'RandomForestClassifier' in algo1:
        if not algo_params:
            clf = RandomForestClassifier(random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            print("training done")
            return clf
        else:
            for u in algo_params:
                algo_tune=u
            if "RandomForestClassifier" in algo_tune:
                var1=[]
                keys1=[]
                for key,value in algo_params[u].items():
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
                print("training done")
                return clf1
    elif 'svm-rbf' in algo1:
        if not algo_params:
            clf=svm.SVC(kernel='rbf',random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            print("training done")
            return clf
        else:
            for u in algo_params:
                algo_tune=u
            if "svm-rbf" in algo_tune:   
                var3=[]
                keys3=[]
                for key,value in algo_params[u].items():
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
                print("training done")
                return clf1
    elif 'svm-linear' in algo1:
        if not algo_params:
            clf=svm.LinearSVC(C=1.0,random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            print("training done")
            return clf
        else:
            for u in algo_params:
                algo_tune=u
            if "svm-linear" in algo_tune:
                var4=[]
                keys4=[]
                for key,value in algo_params[u].items():
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
                print("training done")
                return clf1
    elif 'KNeighborsClassifier' in algo1:
        if not algo_params:
            clf= KNeighborsClassifier(n_neighbors=3,random_state=RANDOM_STATE)
            clf = clf.fit(X_train,y_train)
            print("training done")
            return clf
        else:
            for u in algo_params:
                algo_tune=u
            if "KNeighborsClassifier" in algo_tune:
                var5=[]
                keys5=[]
                for key,value in algo_params[u].items():
                    if value !="":
                        var5.append(key)
                        keys5.append(value)
                if 'n_neighbors' in var5:
                    indi16=var5.index('n_neighbors')
                    n_neighborss=int(keys5[indi16])
                else:
                    n_neighborss=5
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
                clf1=KNeighborsClassifier(n_neighbors=n_neighborss,algorithm=algorithms,p=ps,random_state=RANDOM_STATE)
                clf1 = clf1.fit(X_train,y_train)
                print("training done")
                return clf1
    elif 'GaussianNBclassifier' in algo1:
        clf = GaussianNB()
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'LinearRegressor' in algo1:
        #if not algo_params:
        clf=LinearRegression()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u
    elif 'DecisiontreeRegressor' in algo1:
        #if not algo_params:
        clf=DecisionTreeRegressor()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u
    elif 'LogisticRegressor' in algo1:
        #if not algo_params:
        clf=LogisticRegression()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u
    elif 'RandomforestRegressor' in algo1:
        #if not algo_params:
        clf=RandomForestRegressor(n_estimators =30, random_state = 42)
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u
    elif 'LassoRegressor' in algo1:
        #if not algo_params:
        clf=linear_model.Lasso()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u
    elif 'RidgeRegressor' in algo1:
        #if not algo_params:
        clf=clf = Ridge(alpha=1.0)
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
        # else:
        #     for u in li:
        #         algo_tune=u


   
