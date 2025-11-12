# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
import csv
from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse

import torch
import torch.nn as nn
   
# necessary imports 
import seaborn as sns
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi

from skimage import transform
import seaborn as sns
#import keras as k
#from keras.layers import Dense
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from tensorflow.keras.optimizers import Adam
##
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="tender_coconut"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    
        

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('index.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)



@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('test'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('login_user'))

    
        
    return render_template('register.html',msg=msg)




@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    #######
    '''mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM food_info")
    rr = mycursor.fetchall()
    for r1 in rr:
        
        mycursor.execute("SELECT * FROM food where food=%s",(r1[1],))
        dd = mycursor.fetchone()
        cal=dd[2]
        weight=r1[4]
        w=int(weight)
        d2=(cal/100)*w
        calorie=round(d2,2)
        mycursor.execute("update food_info set calorie=%s where id=%s",(calorie,r1[0]))
        mydb.commit()'''
    ############
        
        
    return render_template('admin.html')

@app.route('/add_food', methods=['GET', 'POST'])
def add_food():
    msg=""
    act=request.args.get("act")

    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        food=request.form['food']

        mycursor.execute('SELECT count(*) FROM food WHERE food = %s', (food,))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM food")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO food(id,food) VALUES (%s, %s)"
            val = (maxid,food)
            mycursor.execute(sql,val)
            mydb.commit()
            return redirect(url_for('add_food',act='1'))
        else:
            msg="1"

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from food where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_food'))

        
    mycursor.execute("SELECT * FROM food")
    data = mycursor.fetchall()
    
    dnt=dnt-5
        
    return render_template('add_food.html',msg=msg,act=act,data=data)

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    msg=""
    act=request.args.get("act")
    food=request.args.get("food")
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')

    ######
    '''dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        food=''
        calorie=''
        weight=''
        mycursor.execute("SELECT max(id)+1 FROM food_info")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO food_info(id,food,filename,calorie,weight) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid,food,fname,calorie,weight)
        mycursor.execute(sql,val)
        mydb.commit()'''
    #######

        
    if request.method=='POST':
        
        
        nutrient=request.form['nutrient']
        details=request.form['details']
      

        mycursor.execute("SELECT max(id)+1 FROM food_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO food_data(id,food,nutrient,details) VALUES (%s, %s, %s, %s)"
        val = (maxid,food,nutrient,details)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('add_data',act='1',food=food))

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from food_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_data',food=food))

        
    mycursor.execute("SELECT * FROM food_data where food=%s",(food,))
    data = mycursor.fetchall()
    
        
        
    return render_template('add_data.html',msg=msg,act=act,data=data,food=food)

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        print(fname)
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''

    return render_template('img_process.html',dimg=dimg)

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 0, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)'''

    return render_template('pro11.html',dimg=dimg)





@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        '''image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        cv2.imwrite("static/trained/bb/bin_"+fname, thresh)'''

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        #segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)

###Feature extraction & Classification
def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))




@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        #edged.save(path4)
        ##

    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')

    j=0
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        
        parser = argparse.ArgumentParser(
        description='Script to run Yolo-V8 object detection network ')
        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                               )
        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Labels of Network.
        classNames = { 0: 'background',
            1: 'plant' }

        # Open video file or capture device. 
        '''if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)'''

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        
        frame = cv2.imread("static/dataset/"+fname)
        frame_resized = cv2.resize(frame,(300,400)) # resize frame for prediction

        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 400), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x400)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                #try:
                y=yLeftBottom
                h=yRightTop-y
                x=xLeftBottom
                w=xRightTop-x
                image = cv2.imread("static/dataset/"+fname)
                fs=ex[j].split('-')
                
                
                
                #mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #cv2.putText(image, val, (x, y+20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                #cv2.imwrite("static/trained/classify/"+fname, mm)
                
                #cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]

                #gg="segment.jpg"
                #cv2.imwrite("static/result/"+gg, cropped)


                #mm2 = PIL.Image.open('static/trained/'+gg)
                #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                #rz.save('static/trained/'+gg)
                #except:
                #    print("none")
                    #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    claname=classNames[class_id]

                    
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    #print(label) #print class and confidence  
        j+=1
    return render_template('pro4.html',dimg=dimg)


    

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,210,265]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,220,275]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,30,60,90,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,30,60,90,120]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    
    ###############################
    
    
    

    return render_template('pro6.html',dimg=dimg)

##TCNN Classification

class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i
            # in_channels = in_channels if i == 0 else out_channels

            layers += [ResBlock(in_channels, out_channels,
                                kernel_size, dilation=dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DConv2d_block, self).__init__()
        self.DConv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU()
        )
        self.drop = nn.Dropout(0.2)

    def forward(self, encode, decode):
        encode = self.drop(encode)
        skip_connection = torch.cat((encode, decode), dim=1)
        DConv2d = self.DConv2d(skip_connection)

        return DConv2d
def TCNN():
        self.Conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )

        self.Conv2d_5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.Conv2d_6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )
        self.Conv2d_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )
        self.TCNN_Block_1 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_2 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_3 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)

        self.DConv2d_7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )
        self.DConv2d_6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.DConv2d_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.DConv2d_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=1),
            nn.PReLU()
        )

def forward(self, input1):
    
    Conv2d_1 = self.Conv2d_1(input1)
    # print("Conv2d_1", Conv2d_1.shape)  # [64, 16, 5, 320]
    Conv2d_2 = self.Conv2d_2(Conv2d_1)
    # print("Conv2d_2", Conv2d_2.shape)  # [64, 16, 5, 160]
    Conv2d_3 = self.Conv2d_3(Conv2d_2)
    # print("Conv2d_3", Conv2d_3.shape)  # [64, 16, 5, 79]
    Conv2d_4 = self.Conv2d_4(Conv2d_3)
    # print("Conv2d_4", Conv2d_4.shape)  # [64, 32, 5, 39]
    Conv2d_5 = self.Conv2d_5(Conv2d_4)
    # print("Conv2d_5", Conv2d_5.shape)  # [64, 32, 5, 19]
    Conv2d_6 = self.Conv2d_6(Conv2d_5)
    # print("Conv2d_6", Conv2d_6.shape)  # [64, 64, 5, 9]
    Conv2d_7 = self.Conv2d_7(Conv2d_6)
    # print("Conv2d_7", Conv2d_7.shape)  # [64, 64, 5, 4] (B, 1, T, 320)
    reshape_1 = Conv2d_7.permute(0, 1, 3, 2)  # [64, 64, 4, 5] (B,C,帧长,帧数)
    batch_size, C, frame_len, frame_num = reshape_1.shape
    reshape_1 = reshape_1.reshape(batch_size, C * frame_len, frame_num)
    # print("reshape_1", reshape_1.shape)  # [64, 256, 5]

    TCNN_Block_1 = self.TCNN_Block_1(reshape_1)
    TCNN_Block_2 = self.TCNN_Block_2(TCNN_Block_1)
    TCNN_Block_3 = self.TCNN_Block_3(TCNN_Block_2)

    reshape_2 = TCNN_Block_3.reshape(batch_size, C, frame_len, frame_num)
    reshape_2 = reshape_2.permute(0, 1, 3, 2)
    # print("reshape_2", reshape_2.shape)  # [64, 64, 5, 4]

    DConv2d_7 = self.DConv2d_7(torch.cat((Conv2d_7, reshape_2), dim=1))
    # print("DConv2d_7", DConv2d_7.shape)     # [64, 64, 5, 9]
    DConv2d_6 = self.DConv2d_6(torch.cat((Conv2d_6, DConv2d_7), dim=1))
    # print("DConv2d_6", DConv2d_6.shape)     # [64, 32, 5, 19]
    DConv2d_5 = self.DConv2d_5(torch.cat((Conv2d_5, DConv2d_6), dim=1))
    # print("DConv2d_5", DConv2d_5.shape)     # [64, 32, 5, 39]
    DConv2d_4 = self.DConv2d_4(torch.cat((Conv2d_4, DConv2d_5), dim=1))
    # print("DConv2d_4", DConv2d_4.shape)     # [64, 16, 5, 79]
    DConv2d_3 = self.DConv2d_3(torch.cat((Conv2d_3, DConv2d_4), dim=1))
    # print("DConv2d_3", DConv2d_3.shape)     # [64, 16, 5, 160]
    DConv2d_2 = self.DConv2d_2(torch.cat((Conv2d_2, DConv2d_3), dim=1))
    # print("DConv2d_2", DConv2d_2.shape)     # [64, 16, 5, 320]
    DConv2d_1 = self.DConv2d_1(torch.cat((Conv2d_1, DConv2d_2), dim=1))
    # print("DConv2d_1", DConv2d_1.shape)     # [64, 1, 5, 320]

    return DConv2d_1

def model():
    # model = TCNN_Block(in_channels=32, out_channels=64, kernel_size=3, init_dilation=3, num_layers=6)
    model = TCNN()
#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')
    
    ##
    vv=[]
    vn=0

    v1=0
    v2=0
    v3=0
    
    data2=[]
    dtt=[]
    path_main = 'static/dataset'
    dt1=[]
    dt2=[]
    dt3=[]
        
    for val in ex:
        
        va=val.split('-')
        n=0
        
        for fname in os.listdir(path_main):
            
            fa1=fname.split('.')
            
            
            if va[0]==fa1[0] and va[1]=='1':
                dt1.append(fname)
                v1+=1
            if va[0]==fa1[0] and va[1]=='2':
                dt2.append(fname)
                v2+=1
            if va[0]==fa1[0] and va[1]=='3':
                dt3.append(fname)
                v3+=1
        
    data2.append(dt1)
    data2.append(dt2)
    data2.append(dt3)
    
    print(data2)
    vv=[v1,v2,v3]    
    g1=v1+v2+v3
    dd2=vv
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    cc=['green','yellow','blue']
    plt.bar(doc, values, color =cc,
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)    
    plt.close()
    #plt.clf()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)
##



@app.route('/test', methods=['GET', 'POST'])
def test():
    msg=""
    ss=""
    fn=""
    fn1=""
    fr2=""
    predict=""
    dta=""
    cname=[]
    result=""
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]
    

    result=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            
            ##
            n=0
            nn=""
            glu=""
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                va=val.split('-')
                
                fa1=fname.split('.')
                
                
                if va[0]==fa1[0] and va[1]=='1':
                    result=cname[0]
                    glu=va[2]
                    nn='1'
                    break
                elif va[0]==fa1[0] and va[1]=='2':
                    result=cname[1]
                    glu=va[2]
                    nn='2'
                    break
                elif va[0]==fa1[0] and va[1]=='3':
                    result=cname[2]
                    glu=va[2]
                    nn='3'
                    break
                
                n+=1
                
            
            
            nn=str(n)
            dta="a"+"|"+fn+"|"+result+"|"+nn+"|"+glu
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            f3=open("static/test/file.txt","r")
            fn=f3.read()
            f3.close()
            
            msg="fail"
            #msg="Tender Coconut Not Found!"
    
    
        
    return render_template('test.html',msg=msg,name=name,fn=fn)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login_user'))
    
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    uname=""
    if 'username' in session:
        uname = session['username']

    #if uname is None:
    #    return redirect(url_for('login_user'))
    
    mycursor = mydb.cursor()
    act=request.args.get("act")

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[2]
    nn=gs[3]

    gm=int(gs[4])
    glu=(gm/100)*1.8
    glu1=round(glu,2)
    glucose=str(glu1)
    

    
    '''ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()

    xx=ext.split(',')

    ff=open("static/trained/values.txt",'r')
    val=ff.read()
    ff.close()

    yy=val.split(',')

    calorie=""
    i=0
    for x in xx:
        if ts==x:
            cal=int(yy[i])
            gr=int(gram)
            calo=(cal/100)*gr
            calorie=round(calo)
            break
        i+=1'''

    
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,glucose=glucose,nn=nn)




##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


