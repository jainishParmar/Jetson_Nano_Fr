import face_recognition as fr
import cv2
import os
import pickle

Encoding=[]
Names=[]
j=0
image_dir='F:\jetson_nano\demoImages-master\known'
for root,dirs,files in os.walk(image_dir):
    for file in files:
        path=os.path.join(root,file)
        name=os.path.splitext(file)[0]
        person=fr.load_image_file(path)
        encoding=fr.face_encodings(person)[0]
        Encoding.append(encoding)
        Names.append(name)


with open('train.pkl','wb') as f:
    pickle.dump(Names,f)
    pickle.dump(Encoding,f)

Encoding=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encoding=pickle.load(f)

font=cv2.FONT_HERSHEY_SIMPLEX
image_dir=r'F:\jetson_nano\demoImages-master\unknown'

for root,dirs,files in os.walk(image_dir):
    for file in files:
        print(root)
        print(file)
        testImagePath=os.path.join(root,file)
        testImage=fr.load_image_file(testImagePath)
        facePositions=fr.face_locations(testImage)  
        allEncodins=fr.face_encodings(testImage,facePositions)
        testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)

        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodins):
            name="unknown person"
            matches=fr.compare_faces(Encoding,face_encoding)
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]
            cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(testImage,name,(left,top-6),font,.75,(255,0,0),2)

            cv2.imshow('pic',testImage)
            cv2.moveWindow('pic',0,0)
            if cv2.waitKey(0)==ord('q'):
                cv2.destroyAllWindows()


