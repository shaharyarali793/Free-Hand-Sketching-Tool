from flask import Flask,render_template,request,redirect,jsonify
# from sklearn.model_selection import train_test_split as tts
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
# from keras.utils import np_utils
# from random import randint
# import numpy as np
# import os
from PIL import Image

# import base64
# # from PIL import Image
# import os
# # import tensorflow as tf
# from random import randint
# import numpy as np
# import cv2
# import json
# # import matplotlib.pyplot as plt
# import random
# import pandas as pd
# # from DeepImageSearch import Index,LoadData,SearchImage
# import flask
import sqlite3
import base64
# from static.p import calculation
# Disabling the status message
# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)


from random import randint
from more_itertools import one
import numpy as np
import os
# from PIL import Image
# import tensorflow as tf
import tensorflow as tf
import cv2
import random

import json



app = Flask(__name__)

model =  tf.keras.models.load_model("./model/parts_prediction(pro).h5")
picfolder = os.path.join('static','suggestions')

app.config['UPLOAD_FOLDER'] = picfolder


# currentlocation = os.path.dirname(os.path.abspath(__file__))

@app.route("/login",methods = ["POST","GET"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # print(username)
        # print(password)

       


        sqlconnection = sqlite3.Connection("database.db")
        cursor = sqlconnection.cursor()

        query1 = "Select username,password from user Where user.username = '{u}' and user.password = '{p}';".format(u=username,p=password)
        rows = cursor.execute(query1)
        rows = rows.fetchall()


        if len(rows) == 1:
            return redirect("/sketch")
        else:
            return redirect("/signup")

@app.route("/signup",methods = ["GET","POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    if request.method == "POST":
            username = request.form['username']
            password = request.form['password']
            email =  request.form['email']

            sqlconnection = sqlite3.Connection("database.db")
            cursor = sqlconnection.cursor()

            query2 = "INSERT INTO user VALUES('{u}','{e}','{p}');".format(u=username,p=password,e=email)
            cursor.execute(query2)
            sqlconnection.commit()

            return redirect("/login")














@app.route("/")
def main():
    file = open("data.txt",'r+')
    file.truncate(0)
    file.close()
    return render_template("welcome.html")




@app.route("/sketch", methods=["GET", "POST"])
def ready():
    if request.method == "GET":
        return render_template("sketch.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        img = base64.b64decode(data)
        with open('sketching/temp.png', 'wb') as output:
            output.write(img)

        #Image Resizing
        img = Image.open("sketching/temp.png")
        newsize = (28, 28)
        im1 = img.resize(newsize)
        # Shows the image in image viewer
        im1.save("sketching/new.png")

        # # #Removing temp.png
        # os.remove("sketching/temp.png")

        #Reading image
        img = cv2.imread("sketching/new.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        probability = prediction[0][2]
        print(prediction)
        # a = np.array(prediction)
        # print(a.shape)


        Decision ={0: "eyes", 1: "lip",2:"nose"}
        # probability = np.argmax(prediction)
        #Making Prediction
        result = Decision[np.argmax(prediction)]
        # print(result)

        # if result == "face":
        #     print("Face is drawn!")
            #Displaying Suggestion
        imageList = os.listdir(f'static/suggestions/{result}_doodle_dataset/')
        random.shuffle(imageList)
        imageList = [f'/suggestions/{result}_doodle_dataset/' + image for image in imageList]
        # print(imageList)
        items = [imageList,result]
        list_lenght = len(imageList)
        one_picture = random.choice(imageList)
        print(one_picture)
        print(probability)

        content = {
            "image":one_picture,
            "Probability":probability,
            "Prediction":result
        }
        # return render_template('sketch.html',imageList =one_picture,pro=probability)
        print(result)
        return render_template('sketch.html',image=one_picture,Probability=probability,Prediction=result)


    # calculation()

    #New Model
    # Importing the proper classes


    # load the Images from the Folder (You can also import data from multiple folders in python list type)
    # image_list = LoadData().from_folder(folder_list=['./static/sketches'])

    # # For Faster Serching we need to index Data first, After Indexing all the meta data stored on the local path
    # Index(image_list).Start()


    # for searching, you need to give the image path and the number of the similar image you want
            # images = SearchImage().get_similar_images(image_path="sketching/temp.png",number_of_images=10)

 

   
 


    #Displaying Suggestion
            # imageL = []
            # for items in images:
            #     imageL.append(f"{items}.png")
            # return render_template('sketch.html',imageList = imageL)
        # else:
        #     print("No Face Drawing Found!")
        #     message = "No Face Drawing Found!"
        #     return render_template('sketch.html',msg = message)


   


  

@app.route('/test', methods=['POST'])
def test():
    # if request.method == "GET":
    #     return render_template("sketch.html")

    # if request.method == "POST":
    output = request.get_json()
    print(output) # This is the output that was stored in the JSON within the browser
    print(type(output))
    result = json.loads(output) #this converts the json output to a python dictionary
    print(result) # Printing the new dictionary
    print(type(result))#this shows the json converted as a python dictionary
    file = open("data.txt","w")
    file.write(result['userInfo'])
        # return render_template("match.html",imf = result)
    return "Information Recieved Successfully"

@app.route("/match",methods=["POST","GET"])
def match():
    filename = open("data.txt",'r+')
    image_path = "./"+filename.read()
    if request.method == "POST":
        new = image_matching(image_path)
        return render_template("match.html",imf =image_path,iff = new  )

    return render_template("match.html",imf =image_path )

    
    
def image_matching(image_1):
    newI = ""
    file = os.listdir("./static/original")
    for i in file:
        original = cv2.imread(image_1)
        duplicate = cv2.imread("./static/original/{k}".format(k=i))# 1) Check if 2 images are equals
        print(original.shape)
        print(duplicate.shape)
        if original.shape == duplicate.shape:
            print("The images have same size and channels")
            difference = cv2.subtract(original, duplicate)
            print(difference)
            b, g, r = cv2.split(difference)
            print(b,g,r)

            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                print("The images are completely Equal")
            return(f"./static/original/{i}")
            

if __name__ == "__main__":
    app.run(debug=True)