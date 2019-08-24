# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:41:58 2019

@author: robin
"""

import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import pandas as pd
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
emal1=[]
nam1=[]
embd1=[]
maxi=[]
max_embed=float(0.0)
title=[['emailid','name','userphoto']]
tf.reset_default_graph()
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160
j=0
sess = tf.Session()
thresholde=.9
k=0

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')
#names=[' ','anirudh',' ','surya','achutha','yogesh','sanyukta','ishita','robin','abishek','saquib','abha','pankhi','stuti','']
names=[]
vecs=[]
#names=[' ','anirudh',' ','surya','achutha',' ',' ',' ',' ',' ',' ',' ',' ',' ','jasper','anand','anuj','skandhan','Aravindraj','Natraj','Ashwin','Jones','Jeffery','deepak']
facenet.load_model("20170512-110547/20170512-110547.pb")
#names=np.load("G:/FaceDetection/facematch/names.npy")


# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    #file=open("em.txt",'w')
    #cont=file.write(np.array_str(embedding))
  #  print(embedding)
    #data={'points':[embedding]}
    #df=pd.DataFrame(data)
    #df.to_csv('neon.csv',index='false')
  
   
    
   # print('File Successfully written.')
    return embedding
    #file.close()
def compare2face(facess):
    
    ind=0
    flag='False'
    for i in range(len(vecs)):
        #print(npys[i].shape,facess.shape)
        dist=np.sqrt(np.sum(np.square(np.subtract(vecs[i], facess))))
        #dist=int(dist)
        if dist<=thresholde:
             ind=i+1
             
             return ind
            #print(dist)    
        # calculate Euclidean distance
            #dist = np.sqrt(np.sum(np.square(np.subtract(i, facess))))
         # ind=i+1
          #   break
    flag='True'
    return flag       
cap = cv2.VideoCapture(1)#"GH011493.mp4")#0)
z=0
mail_list=[]
ret, image = cap.read()
[h, w] = image.shape[:2]
out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))
name_index_counter=0
while(cap.isOpened()):
    ret, frame = cap.read()  
    print('z',z)
    if z%4==0 and ret==True:    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        #img = imutils.resize(frame,width=1000)
        faces = getFace(frame)
    
          
        for face in faces:
            #file=open("em.txt",'w')
            #cont=file.write()
            cv2.rectangle(frame,(face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
            name=compare2face(face['embedding'])
            max_embed=max(face['embedding'])
            if name=='True':
               
                dict={"embd":":emal1,"embedding":embd1,"max_feature":maxi}
                df=pd.DataFrame(dict)
                df.to_csv('dota.csv')                 
                email=input("Enter your email-id:")
                nam=input("Enter your name")
                emal1.append(email)
                nam1.append(nam)
                maxi.append(max_embed)
                
                
                #print((face['embedding'][0]))
                s=sum(face['embedding'][0])
                l=len(face['embedding'][0])
                avg=s/l
                print(l)
                print(s)
                print(avg)
                embd1.append(face['embedding'])
                #print(maxi)
                
                print(emal1)
                print(nam1)
                j=j+1
                vecs.append(face['embedding'])
                names.append(str(name_index_counter))
                mail_list.append(str(email))
                name_index_counter += 1
            else:
                nm=mail_list[name-1]
                email_user = 'shopfinite6@gmail.com'
                email_password = 'wewillwin'
                email_send = nm

                subject = 'Welcome To FiniteShop'

                

                msg = MIMEMultipart()
                msg['From'] = email_user
                msg['To'] = email_send
                msg['Subject'] = subject

                body =  'Check out our newest arrival of apparels, Exciting discounts on Pants, Shirts, Casuals.Latest arrival of electronic goods.'
                msg.attach(MIMEText(body,'plain'))

                filename='finite.jpg'
                attachment  =open(filename,'rb')
                
                print("The total customers till now:")
                print (j)
                part = MIMEBase('application','octet-stream')
                part.set_payload((attachment).read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',"attachment; filename= "+filename)

                msg.attach(part)
                text = msg.as_string()
                try:
                    server = smtplib.SMTP('smtp.gmail.com',587)
                    server.starttls()
                    server.login(email_user,email_password)
                    server.sendmail(email_user,email_send,text)
                    server.quit()
                    print('Done')
                    
                except:
                    print("Failed to send the Email!\nPlease check whether you have entered the name of the file properly")
            
    # =============================================================================
#             if(name==0):
#                 for k in range(0,400):
#                     with open('neon.csv','a') as csvfile :
#                         #  g=embedding[0][0]
#                         colnames=['id', 'embd'] 
#                         writer=csv.writer(csvfile)
#                         writer.writerow([k,face['embedding']])
#                         
#                         data=pd.read_csv("neon.csv")
#                         data_top=data.head(2)
# =============================================================================
            
          #  print(names[name])
            #file.close()
            #cv2.putText(frame, names[name], (face['rect'][0],face['rect'][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA)
       #cv2.imshow('img',faces[0]['face'])
       
        cv2.imshow("faces", frame)    
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    z += 1       
dict={"name":nam1,"email-id":emal1,"embedding":embd1,"max_feature":maxi}
df=pd.DataFrame(dict)
df.to_csv('dota.csv')    
#print([names])
#print([vecs])
cv2.destroyAllWindows()   
cap.release()
print("The total customers in this day:")
print(j)
out.release()         
# =============================================================================
# img = cv2.imread("G://FaceDetection//facematch//images//Esha-gupta.jpg",1)
# #img = imutils.resize(img,width=1000)
# faces = getFace(img)
# for face in faces:
#     cv2.rectangle(img, (face['rect'][0], face['rect'][1]), (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
# cv2.imshow("faces", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

