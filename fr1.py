import cv2
import face_recognition as fr


while True:
    image=fr.load_image_file("F:\jetson_nano\demoImages-master\known\Chuck Schumer.jpg")
    face_locations=fr.face_locations(image)
    print(face_locations)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for (row1,col1,row2,col2) in face_locations:
        cv2.rectangle(image,(col1,row1),(col2,row2),(0,0,255),2)
    cv2.imshow('nanocam',image)
    cv2.moveWindow('nanocam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()





# import images from github
# known image & unkonwn image for trainng  and testing
# set image name to testing



