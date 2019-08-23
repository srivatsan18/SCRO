import numpy as np
import cv2
import pickle
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}
with open("labels.picke",'rb') as f:
	og_labels=pickle.load(f)
	labels={v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0)

while(True):
	#capturing frame-by frame
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
	for(x,y,w,h) in faces:
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]
        
		img_item = "face-img.jpg"
		cv2.imwrite(img_item,roi_gray)
		#recognize?deep learned model 
		id_, conf=recognizer.predict(roi_gray)
		if conf>=80 and conf<=95:
			print(id_)
			print(labels[id_])
			font=cv2.FONT_HERSHEY_SIMPLEX
			name=labels[id_]
			color=(0,255,255)
			stroke=2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
		color=(255,0,0)#BGR
		stroke=2
		cv2.rectangle(frame,(x,y),((x+w),(y+h)),color,stroke)

	#display the resulting frame
	cv2.imshow("frame",frame)
	if cv2.waitKey(20)& 0xFF==ord('q'):
		break

#whenover release thecapture
cap.release()
cv2.destroyAllWindows()
