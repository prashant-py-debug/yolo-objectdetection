import pandas as pd
import numpy as np
import os
import cv2

df = pd.read_csv("resources/coco.names", names = ["classes"])
class_names = df["classes"].to_list()
# print(class_names)

###########################
whT = 320 #this DarNet uses 320x320 images as input

directory = "test"
model_config = "resources/yolov3-320.cfg"
model_weights = "resources/yolov320.weights"
confiThreshold = 0.5
nms_threshold = 0.3
###########################

def findObject(outputs,img):
    """
    takes DarkNet output and img,
    finds and draws a bounding box around the object.
    """
    hT,wT,cT = img.shape
    bbox = []
    classids = []
    confs = []
    for output in outputs:
        for df in output:
            scores = df[5:]
            classId = np.argmax(scores)
            confi_score = scores[classId]
            if confi_score > confiThreshold:
                w,h = int(df[2]*wT) , int(df[3]*hT)
                x,y = int((df[0]*wT)-w/2), int((df[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classids.append(classId)
                confs.append(float(confi_score))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confiThreshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(img,f'{class_names[classids[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # cv2.imwrite("resources/yolo_identified_1.jpg",img)


    



net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def image_reader(path):
    """
    fuction takes the string address as input
    outputs the list of image address
    """

    paths = os.listdir(path)
    images =[]
    for p in paths:
        images.append(os.path.join(path,p))
    return images


img_dir = image_reader(directory)
for i in img_dir:
    img = cv2.imread(i)

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop = False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    findObject(outputs,img)
    img = cv2.resize(img,(540,540),interpolation = cv2.INTER_AREA)
    cv2.imshow("object",img)
    cv2.waitKey(1)

   





