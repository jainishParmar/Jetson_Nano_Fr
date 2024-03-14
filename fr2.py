import face_recognition as fr
import cv2

donFace = fr.load_image_file(r"F:\jetson_nano\demoImages-master\known\Donald Trump.jpg")
donEncode = fr.face_encodings(donFace)[0]

nancyFace = fr.load_image_file(r"F:\jetson_nano\demoImages-master\known\Nancy Pelosi.jpg")
nancyEncode = fr.face_encodings(nancyFace)[0] 

Encodings=[donEncode,nancyEncode]
names=['donald_trump','nancy_helosi']

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    testImage=fr.load_image_file(r"F:\jetson_nano\demoImages-master\unknown\u11.jpg")
    facePositons=fr.face_locations(testImage)
    allEncoding=fr.face_encodings(testImage,facePositons)

    testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
    for (top,right,bottom,left),face_encoding in zip(facePositons,allEncoding):
        
        name="Unknown person"
        matches=fr.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=names[first_match_index]
        cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(testImage, name,(left,top-6),font,.75,(0,0,255),1)
    cv2.imshow('nanocam',testImage)
    cv2.moveWindow('nanocam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()


    
