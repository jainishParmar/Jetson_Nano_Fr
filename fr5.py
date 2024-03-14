import os
import pickle
import face_recognition as fr

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