import face_recognition as fr
import cv2
import os
import pickle

Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)

font=cv2.FONT_HERSHEY_SIMPLEX
cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=.33,fy=.33)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    # facePositons=fr.face_locations(frameRGB,model='cnn')
    
    facePositons=fr.face_locations(frameRGB)

    allEncodings=fr.face_encodings(frameRGB,facePositons)
    for (top,right,bottom,left),face_encoding in zip(facePositons,allEncodings):
        name='unknown Person'
        matches=fr.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
        
        top=top*3
        right=right*3
        left=left*3
        bottom=bottom*3
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(255,0,0),2)

    cv2.imshow('nanocam',frame)
    cv2.moveWindow('nanocam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
