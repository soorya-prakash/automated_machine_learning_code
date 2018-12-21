from flask import Flask, redirect, url_for, request,send_file,Response
from flask import send_file,render_template
from werkzeug.utils import secure_filename
import os
import preprocess_custom
import shutil
import file_creation
import manipulate_file
import preprocess
import json
import accuracy
import evaluate_params
import re
import pandas as pd
from pathlib import Path
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  response.headers.add('Access-Control-Allow-Credentials', 'true')
  return response
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       try: 
           print('hi')
           f = request.files['file']
           email=request.form['email']
           print(f)
           filename, file_extension = os.path.splitext(f.filename)
           print(file_extension)
           if file_extension not in ['.xlsx','.csv']:
               obj1 = {"status":"False","msg":"only .csv and .xlsx can be processed...try again uploading file with .csv and .xlsx extension"}
               json_data1 = json.dumps(obj1)
               return json_data1
           else:
               PATH=os.path.abspath(os.path.dirname(__file__))
               f.save(os.path.join(PATH, secure_filename(f.filename)))
               res,ts,x=file_creation.filecreation()
               shutil.move(r'/home/ubuntu/copy/master_api/{}'.format(f.filename),r'/home/ubuntu/copy/master_api/root_files/{}/{}'.format(res,f.filename))
               file_path=r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(res,f.filename)
               #def inval(df):
               #    df1 = df.dropna(thresh=2, axis=0)
               #    cols=df1.shape
               #    row=cols[0]
               #    return df1.empty,row
               def is_non_zero_file(fpath):
                   if file_extension =='.csv':
                       df=pd.read_csv(fpath,encoding='latin')
                       cols=df.shape
                       row=cols[0]
                       col=cols[1]
                       print("came in")
                #       invalid,row=inval(df)
                       return df.empty,row,col
                   else:
                       df=pd.read_excel(fpath,encoding='latin')
                 #      invalid=inval(df)
                       cols=df.shape
                       print("came inn")
                       row=cols[0]
                       col=cols[1]
                       return df.empty,row,col   
          #fpath=r"C:\Users\sooryaprakash\Desktop\dsm -ai\new dataset27.7\next set\0731-massachusetts-8000.csv"
               ans,row,col=is_non_zero_file(file_path)
               if ans:
                   error={"status":"False","msg":"empty file uploaded!!!!!!"}
                   json_error=json.dumps(error)
                   return json_error
               elif row<=1:
                   err={"status":"False","msg":"uploaded file should have atleast 30 rows to build the model"}
                   json_err=json.dumps(err)
                   return json_err
               elif col<=1:
                   err={"status":"False","msg":"uploaded file should have atleast 2 columns to build the model"}
                   json_err=json.dumps(err)
                   return json_err
               else:      
                   ss =str(re.findall('uid:(.+)',res))
                   s={"filePath":file_path,"fileExtension":file_extension,"modelId":"{}-{}".format(ts,x),"email":email}
                   json_success=json.dumps(s)
                   json_fetch=json.loads(json_success)
                   abspath=json_fetch.get('filePath')
                   ex=json_fetch.get('fileExtension')
                   mid=json_fetch.get('modelId')
                   email=json_fetch.get('email')
                   print(mid) 
                   print(email)
                   result=manipulate_file.parse(abspath,ex,mid,email)
                   print("main function called")
                   return result
       except Exception as e:
           print(e)
           json_fail2={"status":"False","msg":"invalid file... try again with proper file"}
           res2=json.dumps(json_fail2)
           return res2
@app.route('/algo_select',methods = ['POST', 'GET'])
def main1():
  if request.method == 'POST':
    request_json = request.get_json()
    print(request_json)
    try:
        filepath = request_json.get('filePath')
        email=request_json.get('email')
        ex_path=Path(filepath)
        dy_folder=str(ex_path.parts[6])
        ex=str(request_json.get("fileExtension"))
        mid=request_json.get('modelId')
        deploy=request_json.get('deploy')
        ptype=request_json.get('prediction_type')
        x=request_json.get("x_list")
        if len(x)==0:
            json_fail1={"status":"False","msg":"enter x feature"}
            res=json.dumps(json_fail1)
            return res
        y=request_json.get("y_list")
        if len(y)==0:
            json_fail2={"status":"False","msg":"enter y feature"}
            res1=json.dumps(json_fail2)
            return res1
        algo=str(request_json.get("algo"))
        if len(algo)==0:
            json_fail2={"status":"False","msg":"select algorithm"}
            res2=json.dumps(json_fail2)
            return res2
        tt_split=str(request_json.get("tt_split"))
        if len(tt_split)==0:
            json_fail2={"status":"False","msg":"select split ratio"}
            res3=json.dumps(json_fail2)
            return res3
        print(filepath,ex,x,y,algo,tt_split)  
        xlist=list(x.keys())
        ylist=list(y.keys())
        print(xlist,ylist)
        ylast=str(ylist[0])
        yobj=y[ylast]
        print(yobj)
        if ex in ['.csv']:
            df=pd.read_csv(filepath,encoding='latin')
        else:
            df=pd.read_excel(filepath,encoding='latin')
        print(df)
        print(df[ylist[0]].dtypes)
        ytype=str(df[ylist[0]].dtypes)
        val=len(list(df[ylist[0]].unique()))
        if 'int64' in ytype:
            if val==2 or val==3 :
                algos='classification'
        elif val>20:
            algos='regression'
        elif 'object' in ytype and val<=20 :
            algos='classification'
        elif 'int64' or 'float64' in ytype:
            algos='regression'      
        col_list=df.shape
        check_col=col_list[1]
        check_rows=col_list[0]
        if check_col>1:
            if check_rows>=30:
                if "custom" in ptype and deploy=="False":
                    result=accuracy.preprocess_dropna1(filepath,ex,xlist,ylast,algos,tt_split,mid,email,yobj,ptype,deploy)
                    return result
                elif "custom" in ptype and deploy=="True":
                    label_encoder_path=request_json.get('label_encoder_file_path')          
                    print(label_encoder_path)
                    algo_params=request_json.get('algo_params') 
                    print("yes")
                    result=preprocess_custom.preprocess_dropna(filepath,ex,xlist,ylast,algo,tt_split,mid,email,yobj,ptype,deploy,algo_params,val,label_encoder_path)
                    temp_res=json.loads(result)
                    print('done')
                    if temp_res['status']!='False':
                        temp_pkl=temp_res['pklfile']
                        #temp_result=temp_res['result']
                        shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl))                                       
                        #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_result))
                        add =json.loads(result)
                        add['xlist'] = x
                        add['pklfile']=r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)
                        result=json.dumps(add)
                        return result
                    else:
                        return result
                elif "auto" in ptype and 'False' in deploy:
                    print("started")
                    result=accuracy.preprocess_dropna_ai(filepath,ex,xlist,ylast,algos,tt_split,mid,email,yobj,ptype,deploy)
                    print("fininshed")
                    result_json_parser =json.loads(result)
                    label_encoder_path=result_json_parser['label_encoder_file_path']
                    algo_to_build=result_json_parser['accuracy']
                    final_algo=[]
                    for key in algo_to_build:
                        final_algo.append(key) 
                    print(final_algo)            
                    algo=final_algo
                    filepath = request_json.get('filePath')                             
                    email=request_json.get('email')                                              
                    ex_path=Path(filepath)
                    dy_folder=str(ex_path.parts[6])      
                    ex=str(request_json.get("fileExtension"))
                    mid=request_json.get('modelId')
                    deploy=request_json.get('deploy')
                    ptype=request_json.get('prediction_type')
                    x=request_json.get("x_list")
                    if len(x)==0:
                        json_fail1={"status":"False","msg":"enter x feature"}
                        res=json.dumps(json_fail1)
                        return res
                    y=request_json.get("y_list")
                    if len(y)==0:
                        json_fail2={"status":"False","msg":"enter y feature"}
                        res1=json.dumps(json_fail2)
                        return res1
                    tt_split=str(request_json.get("tt_split"))
                    if len(tt_split)==0:
                        json_fail2={"status":"False","msg":"select split ratio"}
                        res3=json.dumps(json_fail2)
                        return res3
                    print(filepath,ex,x,y,algo,tt_split)  
                    xlist=list(x.keys())
                    ylist=list(y.keys())
                    print(xlist,ylist)
                    ylast=str(ylist[0])
                    yobj=y[ylast]
                    print(yobj)
                    if ex in ['.csv']:
                        df=pd.read_csv(filepath,encoding='latin')
                    else:
                        df=pd.read_excel(filepath,encoding='latin')
                    print(df)
        #tar=str(ylist[0])
                    print(df[ylist[0]].dtypes)
                    ytype=str(df[ylist[0]].dtypes)
        #target=df[ylist]
                    val=len(list(df[ylist[0]].unique()))
                    if 'int64' in ytype:
                        if val==2 or val==3 :
                            algos='classification'
                    elif val>20:
                        algos='regression'
                    elif 'object' in ytype and val<=20 :
                        algos='classification'
                    elif 'int64' or 'float64' in ytype:
                        algos='regression'  
                    print("real application going to execute")
                    result=preprocess.preprocess_dropna(filepath,ex,xlist,ylast,algo,tt_split,mid,email,yobj,ptype,deploy,label_encoder_path)
                    temp_res=json.loads(result)
                    if temp_res['status']!='False':
                        temp_pkl=temp_res['pklfile']
                        #temp_result=temp_res['result']
                        shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl))
                        #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_result))
                        add =json.loads(result)
                        add['xlist'] = x
                        add['pklfile']=r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)
                        result=json.dumps(add)
                        return result
                    else:
                        return result         
            else:
                json_fail2={"status":"False","msg":"your dataset should have atleast 30 records to build a model"}
                res3=json.dumps(json_fail2)
                return res3
        else:

            json_fail2={"status":"False","msg":"your dataset should have atleast 2 columns to build the model"}
            res3=json.dumps(json_fail2)
            return res3
    except Exception as e:
        print(e)
        json_fail2={"status":"False","msg":"try again later"}
        res2=json.dumps(json_fail2)
        return res2
@app.route('/evaluate',methods = ['POST', 'GET'])
def main2():
  if request.method == 'POST':
    request_json = request.get_json()
    try:
        filepath = request_json.get('filePath')
        email=request_json.get('email')
        ex_path=Path(filepath)
        dy_folder=str(ex_path.parts[6])
        ex=str(request_json.get("fileExtension"))
        mid=request_json.get('modelId')
        x=request_json.get("x_list")
        if len(x)==0:
            json_fail1={"status":"False","msg":"enter x feature"}
            res=json.dumps(json_fail1)
            return res
        y=request_json.get("y_list")
        if len(y)==0:
            json_fail2={"status":"False","msg":"enter y feature"}
            res1=json.dumps(json_fail2)
            return res1
        algo=str(request_json.get("algo"))
        if len(algo)==0:
            json_fail2={"status":"False","msg":"error"}
            res2=json.dumps(json_fail2)
            return res2
        tt_split=str(request_json.get("tt_split"))
        if len(tt_split)==0:
            json_fail2={"status":"False","msg":"select split ratio"}
            res3=json.dumps(json_fail2)
            return res3
        algo_params=dict(request_json.get("algo_params"))
        label_encoder_file_path=str(request_json.get("label_encoder_file_path"))
        print(algo_params)
        print(filepath,ex,x,y,algo,tt_split)  
        xlist=list(x.keys())
        ylist=list(y.keys())
        ylast=str(ylist[0])
        yobj=y[ylast]
        print(yobj)
        if ex in ['.csv']:
            df=pd.read_csv(filepath,encoding='latin')
        else:
            df=pd.read_excel(filepath,encoding='latin')
        print(df)
        col_list=df.shape
        check_col=col_list[1]
        check_rows=col_list[0]
        if check_col>1:
            if check_rows>=30:
                result=evaluate_params.preprocess_dropna2(filepath,ex,xlist,ylast,algo,tt_split,mid,algo_params,email,label_encoder_file_path)
                #temp_res=json.loads(result)
        #         if temp_res['status']!='false':
        #             temp_pkl=temp_res['pklfile']
        #             temp_result=temp_res['result']
        # #print(dy_folder)
        #             shutil.move(r"C:\Users\sooryaprakash\Desktop\lt\api\{}".format(temp_pkl),r"C:\Users\sooryaprakash\Desktop\lt\api\root_files\{}\{}".format(dy_folder,temp_pkl))
        #             #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/$
        #             #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),$
        #             shutil.move(r"C:\Users\sooryaprakash\Desktop\lt\api\{}".format(temp_result),r"C:\Users\sooryaprakash\Desktop\lt\api\root_files\{}\{}".format(dy_folder,temp_result))
        #             add =json.loads(result)
        #             add['xlist'] = x
        #             add['pklfile']=r"C:\Users\sooryaprakash\Desktop\lt\api\root_files\{}\{}".format(dy_folder,temp_pkl)
        #          #add['pklfile']="/home/ubuntu/copy/master_api/root_files/{}/{}".form$
        #             result=json.dumps(add)
        #             return result
        #         else:
                return result
      
            else:
                json_fail2={"status":"False","msg":"your dataset should have atleast 30 records to build a model"}
                res3=json.dumps(json_fail2)
                return res3
        else:
            json_fail2={"status":"False","msg":"your dataset should have atleast 2 columns to build the model"}
            res3=json.dumps(json_fail2)
            return res3
   #print("function triggered")
    #if result==NULL:
    #    return "error"
        #temp_res=json.loads(result)
       # if temp_res['status']!='false': 
    #print(temp_res['filePath']) 
            #temp_pkl=temp_res['pklfile']
           # temp_result=temp_res['result']
        #print(dy_folder)
            #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)) 
            #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_result))
            #add =json.loads(result)
           # add['xlist'] = x
          #  add['pklfile']="/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)
         #   result=json.dumps(add)
        #    return result
       # else:
          #  return result
    except Exception as e:
        print(e)
        json_fail2={"status":"False","msg":"try again later"}
        res2=json.dumps(json_fail2)
        return res2
@app.route('/preprocessing_modelling',methods = ['POST', 'GET'])
def main():
  if request.method == 'POST':
    request_json = request.get_json()
    print("hai")
    try:
        filepath = request_json.get('filePath')
        email=request_json.get('email')
        ex_path=Path(filepath)
        dy_folder=str(ex_path.parts[6])
        ex=str(request_json.get("fileExtension"))
        mid=request_json.get('modelId')
        x=request_json.get("x_list")
        type=request_json.get("prediction_type")
        if len(x)==0:
            json_fail1={"status":"False","msg":"enter x feature"}
            res=json.dumps(json_fail1)
            return res
        y=request_json.get("y_list")
        if len(y)==0:
            json_fail2={"status":"False","msg":"enter y feature"}
            res1=json.dumps(json_fail2)
            return res1
        algo=str(request_json.get("algo"))
        if len(algo)==0:
            json_fail2={"status":"False","msg":"select algorithm"}
            res2=json.dumps(json_fail2)
            return res2
        tt_split=str(request_json.get("tt_split"))
        if len(tt_split)==0:
            json_fail2={"status":"False","msg":"select split ratio"}
            res3=json.dumps(json_fail2)
            return res3
        print(filepath,ex,x,y,algo,tt_split)  
        xlist=list(x.keys())
        ylist=list(y.keys())
        ylast=str(ylist[0])
        yobj=y[ylast]
        print(yobj)
        if ex in ['.csv']:
            df=pd.read_csv(filepath,encoding='latin')
        else:
            df=pd.read_excel(filepath,encoding='latin')
        print(df)
        col_list=df.shape
        check_col=col_list[1]
        check_rows=col_list[0]
        if check_col>1:
            if check_rows>=30:
                result=preprocess.preprocess_dropna(filepath,ex,xlist,ylast,algo,tt_split,mid,email,yobj,type)
                temp_res=json.loads(result)
                if temp_res['status']!='False':
                    temp_pkl=temp_res['pklfile']
                    temp_result=temp_res['result']
        #print(dy_folder)
                    shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl))
                    #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/$
                    #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),$
                    shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_result))
                    add =json.loads(result)
                    add['xlist'] = x
                    add['pklfile']="/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)
                 #add['pklfile']="/home/ubuntu/copy/master_api/root_files/{}/{}".form$
                    result=json.dumps(add)
                    return result
                else:
                    return result
      
            else:
                json_fail2={"status":"False","msg":"your dataset should have atleast 30 records to build a model"}
                res3=json.dumps(json_fail2)
                return res3
        else:
            json_fail2={"status":"False","msg":"your dataset should have atleast 2 columns to build the model"}
            res3=json.dumps(json_fail2)
            return res3
   #print("function triggered")
    #if result==NULL:
    #    return "error"
        #temp_res=json.loads(result)
       # if temp_res['status']!='false': 
    #print(temp_res['filePath']) 
            #temp_pkl=temp_res['pklfile']
           # temp_result=temp_res['result']
        #print(dy_folder)
            #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_pkl),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)) 
            #shutil.move(r"/home/ubuntu/copy/master_api/{}".format(temp_result),r"/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_result))
            #add =json.loads(result)
           # add['xlist'] = x
          #  add['pklfile']="/home/ubuntu/copy/master_api/root_files/{}/{}".format(dy_folder,temp_pkl)
         #   result=json.dumps(add)
        #    return result
       # else:
          #  return result
    except Exception as e:
        print(e)
        json_fail2={"status":"False","msg":"try again later"}
        res2=json.dumps(json_fail2)
        return res2
@app.route('/predict',methods = ['POST']) 
def predict():
  if request.method =='POST':
    try:
        predict_json=request.get_json()
        print(predict_json)
    #filepath=predict_json.get('filePath')
        pkl_file=predict_json.get('pklfile')
        print("hi")
        email=predict_json.get('email')
        col_sel=predict_json.get('cols')
        mid=predict_json.get('modelId')
        label_encoder_path=str(predict_json.get('label_encoder_file_path'))
        print(label_encoder_path,mid,email,col_sel,pkl_file)
        y=predict_json.get('y_list')
        ylist=list(y.keys())
        col_list=list(col_sel.keys())
        test_dict=list(predict_json.get('dict'))
        print(test_dict,label_encoder_path)
        if test_dict is None:
            json_fail1={"status":"False","msg":"enter the required fields to predict"}
            res=json.dumps(json_fail1)
            return res    
    #print("data_sent")   
        else:
            try:
                print('aflsjdflkjfdlj')
                predict_res=preprocess.prediction(pkl_file,test_dict,col_list,mid,email,label_encoder_path,ylist)
                print("dooon",predict_res)
                return predict_res
            except:
                json_fail2={"status":"False","msg":"try again later"}
                res2=json.dumps(json_fail2)
                return res2
    except Exception as e:
        json_fail2={"status":"False","msg":"try again later"}
        res2=json.dumps(json_fail2)
        return res2
@app.route("/download")
def hello():
     return '''
       <html><body>
        <a href="/getfile">Click me.</a>
       </body></html>
       '''
@app.route("/getfile",methods=['GET'])
def getPlotCSV():
    if request.method =='GET':
        download=request.args.get('download')
        with open(download) as fp:
            csv = fp.read()
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition":
                      "attachment; filename=result2018-08-14_13-34-13.csv"})
if __name__=='__main__':
    app.run("0.0.0.0",debug=False)
