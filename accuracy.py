from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn import model_selection
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
from sklearn.metrics import accuracy_score
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
def preprocess_dropna1(filepath,ex,x,y,algo,tt_split,mid,email,yobj,type,deploy):     
    global dummy,filepath1,feature_count,sel_x,target_df,comman_list,tt
    tt.append("80% train and 20% test")
    sel_x=x
    print("given x features",x)

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
            pkl_label=encoding(required_df,filepath)
        #normalize()
        #print(list_x)
        #print(list_y)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            print(x1)
            data_spilt(x1,y1)
            print('data split over')
            result_json=model(algo,mid,email,type,deploy,pkl_label)
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
            pkl_label=encoding(required_df,filepath)
            #normalize()
            print(target_df)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            data_spilt(x1,y1)
            result_json=model(algo,mid,email,type,deploy,pkl_label)
            return result_json
def visual():
    print(dummy)
def encode(df_x):
    dictt={}
    label={}
    for i in df_x.select_dtypes(['object']):
        dictt[i]=list(df_x[i].unique())
    for key,val in dictt.items():
        dictt[key]=dict(enumerate(val))
    for i in df_x.select_dtypes(['object']):
        if i in dictt.keys():
            temp =dictt.get(i)
            my_dict2= {y:x for x,y in temp.items()}
            label[i]=dict(my_dict2)
            df_x[i] = df_x[i].map(my_dict2) 
            print("encoded" ,i)
    print(label)
    print(df_x)
    return df_x,label
def encoding(required_df,filepath):
    global dummy,target_df
    global label_encoder,label_encoder1
    ex_path=Path(filepath)
    dy_folder=str(ex_path.parts[6])
   # required_df=required_df.iloc[:,0:-1]
    # for c in required_df.select_dtypes(['object']):
    #     print ("Encoding column " + c)
    #     required_df[c] = label_encoder.fit_transform(required_df[c].astype(str))
    # le=label_encoder.classes_
    # print(le)
    req_df,le=encode(required_df)
    pkl_label =str(r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,"label_encoder.pkl"))
    with open(pkl_label, 'wb') as file:
        pickle.dump(le, file)
    print("executed encoding")  
    # req=required_df
    # req=required_df.iloc[:,0:-1]
    #for c in required_df.select_dtypes(['object']):
        #print ("Encoding column " + c)
        #required_df[c] = label_encoderx.fit_transform(required_df[c].astype(str))
        #np.save('classes.npy',label_encoderx.classes_)
    #for i in target_df.columns:
    #    target_df[i]=label_encoder1.fit_transform(target_df[i].astype(str))
        #np.save('classes.npy', label_encoder1.classes_)
    #ley=label_encoder1.classes_
    #print(ley)
    #print("now")
    #pkl_labely =str(r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,"labely.pkl"))
    #with open(pkl_labely, 'wb') as file:
    #    pickle.dump(ley, file)
    dummy=req_df
    print(dummy)
    return pkl_label
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
def model(algo,mid,email,type,deploy,pkl_label):
    global X_train,X_test,y_train,y_test,filepath1,feature_count,sel_x,dummy,tt
    classifier=['RandomForestClassifier','svm-rbf','svm-linear','KNeighborsClassifier','GaussianNBclassifier','DecisionTreeClassifier']
    regressor=['LinearRegressor','DecisiontreeRegressor','LogisticRegressor','RandomforestRegressor','LassoRegressor','RidgeRegressor']
    frr={}
    if 'regression' in algo:
        if 'LinearRegressor' in regressor:
            clf1=LinearRegression()
            clf1=clf1.fit(X_train,y_train)
            pred1=clf1.predict(X_test)
            print("training done for linearregression")
            acc1=accuracy_score(y_test,pred1)
            frr['LinearRegressor']=acc1
        if 'DecisiontreeRegressor' in regressor:
            clf2=DecisionTreeRegressor()
            clf2=clf2.fit(X_train,y_train)
            pred2=clf2.predict(X_test)
            print("training done for decisiontreeregression")
            acc2=accuracy_score(y_test,pred2)
            frr['DecisionTreeRegressor']=acc2
        if 'LogisticRegressor' in regressor:
            clf3=LogisticRegression()
            clf3=clf3.fit(X_train,y_train)
            pred3=clf3.predict(X_test)
            print("training done for logisticregression")
            acc3=accuracy_score(y_test,pred3)
            frr['LogisticRegressor']=acc3
        if 'RandomforestRegressor' in regressor:
            clf4=RandomForestRegressor()
            clf4=clf4.fit(X_train,y_train)
            pred4=clf4.predict(X_test)
            print("training done for randomforestregression")
            acc4=accuracy_score(y_test,pred4)
            frr['RandomforestRegressor']=acc4
        if 'LassoRegressor' in regressor:
            clf5=linear_model.Lasso()
            clf5=clf5.fit(X_train,y_train)
            pred5=clf5.predict(X_test)
            print("training done for lassoregression")
            acc5=accuracy_score(y_test,pred5)
            frr['LassoRegressor']=acc5
        if 'RidgeRegressor' in regressor:
            clf6 = Ridge(alpha=1.0)
            clf6=clf6.fit(X_train,y_train)
            pred6=clf6.predict(X_test)
            print("training done for ridgeregression")
            acc6=accuracy_score(y_test,pred6)
            frr['RidgeRegressor']=acc6
        ss={"modelId":mid,'accuracy':frr,"label_encoder_file_path":pkl_label,"email":email,"status":"True","algo":algo,"prediction_type":type,'deploy':deploy}
        success_json=json.dumps(ss)
        return success_json

    elif 'classification' in algo:
        if 'DecisionTreeClassifier' in classifier:
            clf7 = tree.DecisionTreeClassifier(random_state=10)
            clf7 = clf7.fit(X_train,y_train)
            pred7=clf7.predict(X_test)
            print("training done for decisiontreeclassifier")
            acc7=accuracy_score(y_test,pred7)
            frr['DecisionTreeClassifier']=acc7
        if 'RandomForestClassifier' in classifier:
            clf8 = RandomForestClassifier( random_state=10)
            clf8 = clf8.fit(X_train,y_train)
            pred8=clf8.predict(X_test)
            print("training done for randomforestclassifier")
            acc12=accuracy_score(y_test,pred8)
            frr['RandomForestClassifier']=acc12
        if 'svm-rbf' in classifier:
            clf9=svm.SVC(kernel='rbf', gamma=0.7, C=1.0,random_state=10)
            clf9 = clf9.fit(X_train,y_train)
            pred9=clf9.predict(X_test)
            print("training done for svm-rbf")
            acc8=accuracy_score(y_test,pred9)
            frr['svm-rbf']=acc8
        if 'svm-linear' in classifier:
            clf10=svm.LinearSVC(C=1.0,random_state=10)
            clf10 = clf10.fit(X_train,y_train)
            pred10=clf10.predict(X_test)
            print("training done for svm-linear")
            acc9=accuracy_score(y_test,pred10)
            frr['svm-linear']=acc9
        if 'KNeighborsClassifier' in classifier:
            clf11= KNeighborsClassifier(n_neighbors=4)
            clf11 = clf11.fit(X_train,y_train)
            pred11=clf11.predict(X_test)
            print("training done for kneighborsclassifier")
            acc10=accuracy_score(y_test,pred11)
            frr['KNeighborsClassifier']=acc10
        if 'GaussianNBclassifier' in classifier:
            clf12 = GaussianNB()
            clf12 = clf12.fit(X_train,y_train)
            pred11=clf12.predict(X_test)
            print("training done for gaussiannbclassifier")
            acc11=accuracy_score(y_test,pred11)
            frr['GaussianNBclassifier']=acc11
        ss={"modelId":mid,'accuracy':frr,"email":email,"status":"True","label_encoder_file_path":pkl_label,"algo":algo,"prediction_type":type,'deploy':deploy}
        success_json=json.dumps(ss)
        return success_json
def preprocess_dropna_ai(filepath,ex,x,y,algo,tt_split,mid,email,yobj,type,deploy):     
    global dummy,filepath1,feature_count,sel_x,target_df,comman_list,tt
    tt.append("80% train and 20% test")
    sel_x=x
    print(x)
    print(y)
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
            pkl_label=encoding(required_df,filepath)
        #normalize()
        #print(list_x)
        #print(list_y)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            print(x1)
            data_spilt(x1,y1)
            result_json=model_ai(algo,mid,email,type,deploy,pkl_label)
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
            pkl_label=encoding(required_df,filepath)
            #normalize()
            print(target_df)
            x1=dummy.iloc[:,0:feature_count]
            y1=dummy.iloc[:,-1]
            data_spilt(x1,y1)
            result_json=model_ai(algo,mid,email,type,deploy,pkl_label)
            return result_json
def model_ai(algo,mid,email,type,deploy,pkl_label):
    global X_train,X_test,y_train,y_test,filepath1,feature_count,sel_x,dummy,tt
    classifier=['RandomForestClassifier','svm-rbf','svm-linear','KNeighborsClassifier','GaussianNBclassifier','DecisionTreeClassifier']
    regressor=['LinearRegressor','DecisiontreeRegressor','LogisticRegressor','RandomforestRegressor','LassoRegressor','RidgeRegressor']
    if 'regression' in algo:
        frr={}
        sorted_algo_accuracy={}
        if 'LinearRegressor' in regressor:
            clf=LinearRegression()
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for linearregression")
            acc1=accuracy_score(y_test,pred)
            frr['LinearRegressor']=acc1
        if 'DecisiontreeRegressor' in regressor:
            clf=DecisionTreeRegressor()
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for decisiontreeregression")
            acc2=accuracy_score(y_test,pred)
            frr['DecisionTreeRegressor']=acc2
        if 'LogisticRegressor' in regressor:
            clf=LogisticRegression()
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for logisticregression")
            acc3=accuracy_score(y_test,pred)
            frr['LogisticRegressor']=acc3
        if 'RandomforestRegressor' in regressor:
            clf=RandomForestRegressor()
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for randomforestregression")
            acc4=accuracy_score(y_test,pred)
            frr['RandomforestRegressor']=acc4
        if 'LassoRegressor' in regressor:
            clf=linear_model.Lasso()
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for lassoregression")
            acc5=accuracy_score(y_test,pred)
            frr['LassoRegressor']=acc5
        if 'RidgeRegressor' in regressor:
            clf = Ridge(alpha=1.0)
            clf=clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for ridgeregression")
            acc6=accuracy_score(y_test,pred)
            frr['RidgeRegressor']=acc6
        rank=sorted(frr, key=frr.get,reverse=True)
        print(rank)
        for g in rank:
            if g in frr:
                sorted_algo_accuracy[g]=frr.get(g)
            break
        print(sorted_algo_accuracy)
        ss={"modelId":mid,"deploy":deploy,"label_encoder_file_path":pkl_label,'accuracy':sorted_algo_accuracy,"email":email,"status":"True","algo":algo,"prediction_type":type}
        success_json=json.dumps(ss)
        return success_json

    elif 'classification' in algo:
        frr={}
        sorted_algo_accuracy={}
        if 'DecisionTreeClassifier' in classifier:
            clf = tree.DecisionTreeClassifier(random_state=10)
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for decisiontreeclassifier")
            acc7=accuracy_score(y_test,pred)
            frr['DecisionTreeClassifier']=acc7
        if 'RandomForestClassifier' in classifier:
            clf = RandomForestClassifier( random_state=10)
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for randomforestclassifier")
            acc12=accuracy_score(y_test,pred)
            frr['RandomForestClassifier']=acc12
        if 'svm-rbf' in classifier:
            clf=svm.SVC(kernel='rbf', gamma=0.7, C=1.0,random_state=10)
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for svm-rbf")
            acc8=accuracy_score(y_test,pred)
            frr['svm-rbf']=acc8
        if 'svm-linear' in classifier:
            clf=svm.LinearSVC(C=1.0,random_state=10)
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for svm-linear")
            acc9=accuracy_score(y_test,pred)
            frr['svm-linear']=acc9
        if 'KNeighborsClassifier' in classifier:
            clf= KNeighborsClassifier(n_neighbors=3)
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for kneighborsclassifier")
            acc10=accuracy_score(y_test,pred)
            frr['KNeighborsClassifier']=acc10
        if 'GaussianNBclassifier' in classifier:
            clf = GaussianNB()
            clf = clf.fit(X_train,y_train)
            pred=clf.predict(X_test)
            print("training done for gaussiannbclassifier")
            acc11=accuracy_score(y_test,pred)
            frr['GaussianNBclassifier']=acc11
        rank=sorted(frr, key=frr.get,reverse=True)
        print(rank)
        for g in rank:
            if g in frr:
                sorted_algo_accuracy[g]=frr.get(g)
            break
        ss={"modelId":mid,"deploy":deploy,'accuracy':sorted_algo_accuracy,"label_encoder_file_path":pkl_label,"email":email,"status":"True","algo":algo,"prediction_type":type}
        success_json=json.dumps(ss)
        return success_json

