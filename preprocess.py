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
def preprocess_dropna(filepath,ex,x,y,algo,tt_split,mid,email,yobj,type,deploy,label_encoder_path):     
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
            visual()
            encoding(required_df,label_encoder_path)
        #normalize()
        #print(list_x)
        #print(list_y)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            print(x1)
            data_spilt(x1,y1)
            result_json=model(algo,email,yobj,ex_dict,mid,type,filepath,deploy,label_encoder_path)
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
            #print(target_df)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            data_spilt(x1,y1)
            result_json=model(algo,email,yobj,ex_dict,mid,type,filepath,deploy,label_encoder_path)
            return result_json
def preprocess_nanimpute():
    global dummy
    df_preprocess=pd.read_csv(path)
    list_x=['Income','Number of Vehicles','Median House Value','Auto Make', 'Auto Model', 'Auto Year', 'Auto Fuel Type', 'Auto MFG', 'Auto Style']
    list_y=['Auto Class']
    comman_list=list_x+list_y
    required_df=df_preprocess[comman_list]
    l=required_df.filter(["Income", "Number of Vehicles","Median House Value","Auto Make","Auto Model","Auto Year","Auto Fuel Type","Auto MFG","Auto Style","Auto Class"]).mode()
    required_df[["Income", "Number of Vehicles","Median House Value","Auto Make","Auto Model","Auto Year","Auto Fuel Type","Auto MFG","Auto Style","Auto Class"]]=required_df[["Income", "Number of Vehicles","Median House Value","Auto Make","Auto Model","Auto Year","Auto Fuel Type","Auto MFG","Auto Style","Auto Class"]].fillna(value=l.iloc[0])
    #required_df[["Income", "Number of Vehicles","Median House Value","Auto Make","Auto Model","Auto Year","Auto Fuel Type","Auto MFG","Auto Style","Auto Class"]].fillna(value=l.iloc[0])
    print(required_df.isnull().sum())
    dummy=dummy.append(required_df)
    visual()
    encoding(required_df)
    #normalize()
    x=dummy[['Income','Number of Vehicles','Median House Value','Auto Make', 'Auto Model', 'Auto Year', 'Auto Fuel Type', 'Auto MFG', 'Auto Style']]
    y=dummy['Auto Class']
    data_spilt(x,y)
    model()
    success={"filePath":filepath,}
    return "SUCCESSFULLY PREPROCESSING OF GIVEN DATA IS DONE AND READY TO FIT THE MODEL"
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
    dummy=req_df
# def normalize():
#     global dummy
#     scaler = StandardScaler()
#     scaled_features=scaler.fit_transform(dummy)
#     scaled_features_df = pd.DataFrame(scaled_features, index=dummy.index, columns=dummy.columns)
#     dummy =scaled_features_df
#     visual()
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
def model(algo,email,obj,ex_dict,mid,type,filepath,deploy,label_encoder_path):
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
    clf=algo_select(algo)
    ex_path=Path(filepath1)
    fp1=str(ex_path.parts[6])
    name=str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if "int64" in obj:
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
    #     res_inverse=inve(Ypredict,y_test)
    # elif "float64" in obj:
    #     res_inverse=inve(Ypredict,y_test)
    # else:
    #     res_inverse=inverse(Ypredict,y_test)
    # ss={"status":"True","deploy":deploy,"prediction_type":type,"model_file_path":filepath,"modelId":mid,"download":"/home/ubuntu/copy/master_api/root_files/{}/{}".format(fp1,res_inverse),"pklfile":pkl_filename,"result":res_inverse,"accuracy":Testscore,"msg":"successfully model created","train_split":train_split,"test_split":test_split,"algo":algo,"email":email,"hint":ex_dict}
    ss={"status":"True","deploy":deploy,"prediction_type":type,"label_encoder_file_path":label_encoder_path,"model_file_path":filepath,"modelId":mid,"pklfile":pkl_filename,"accuracy":Testscore,"msg":"successfully model created","train_split":train_split,"test_split":test_split,"algo":algo,"email":email,"hint":ex_dict}
    success_json=json.dumps(ss)
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
def transform_input_df(input_df,le):
    print("before inverse transforming")
    print(input_df)
    #with open(lable_encoder,'rb') as file:
    #    le=pickle.load(file)
    length_encoded_columns=len(list(input_df.columns))
    if length_encoded_columns != 0 and len(input_df.index) != 0:
        for i in input_df.columns:
            if i in le:
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
def inverse_transform_y(y_label,Ypredict,le):
    print("hi")
    print(le)
    if y_label[0] in le:
        print('soorya')
        temp=le.get(y_label[0])
        mydict={y:x for x,y in temp.items()}
        print(mydict)
        if Ypredict[0] in mydict:
            result =mydict.get(Ypredict[0])
            print(result)
            return result
    else:
        print("y is not in label_encoder")
def prediction(pkl,record,col,mid,email,label_encoder_path,y):
    global dummy,test_df
    print('im here')
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
    print(Ypredict)
    ss = re.findall('model(.+)2018', s)
    print(ss)
    if "(classification)" in ss:
        key=y
        print(key)
        predicted_result=inverse_transform_y(key,Ypredict,le)
        print(predicted_result)
        predicted=str(predicted_result)
        ss={"status":"True","result":predicted,"modelId":mid,"email":email}
        json_result=json.dumps(ss)
        return str(json_result)
    else:
        predicted=str(Ypredict)
        ss={"status":"True","result":predicted,"modelId":mid,"email":email}
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
def algo_select(algo1):
    global X_train,y_train
    if 'DecisionTreeClassifier' in algo1:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'RandomForestClassifier' in algo1:
        clf = RandomForestClassifier(n_estimators=6, max_depth=None,min_samples_split=2, random_state=0)
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'svm-rbf' in algo1:
        clf=svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'svm-linear' in algo1:
        clf=svm.LinearSVC(C=1.0)
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'KNeighborsClassifier' in algo1:
        clf= KNeighborsClassifier(n_neighbors=3)
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'GaussianNBclassifier' in algo1:
        clf = GaussianNB()
        clf = clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'LinearRegressor' in algo1:
        clf=LinearRegression()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'DecisiontreeRegressor' in algo1:
        clf=DecisionTreeRegressor()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'LogisticRegressor' in algo1:
        clf=LogisticRegression()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'RandomforestRegressor' in algo1:
        clf=RandomForestRegressor(n_estimators =30, random_state = 42)
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'LassoRegressor' in algo1:
        clf=linear_model.Lasso()
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf
    elif 'RidgeRegressor' in algo1:
        clf=clf = Ridge(alpha=1.0)
        clf=clf.fit(X_train,y_train)
        print("training done")
        return clf


   
