import numpy as np
import cv2
import imutils
import os
import time
import math

os.environ['DISPLAY'] = ':1'

labelsPath = "../configs/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "../weights/yolov3.weights"
configPath = "../configs/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2)
            * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False


def loadFMDYOLO():
    weightsPath = "../weights/yolov3_fm.weights"
    configPath = "../configs/yolov3_fm.cfg"
    objnamePath = "../configs/fm.names"
    net = cv2.dnn.readNet(weightsPath, configPath)
    classes = []
    with open(objnamePath, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def Setup(yolo):
    global net, ln, LABELS
    weights = "/home/sahg4n/Documents/yolo stuff/yolov3.weights"
    config = configPath
    LABELS = open(labelsPath).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'Masked':
                boxColor = (255, 255, 0)
            else:
                boxColor = (255, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), boxColor, 2)
            (width, height), baseline = cv2.getTextSize(
                label + " {:.2f}".format(confs[0]), font, 1, 1)
            cv2.rectangle(img, (x, y), (x + width, y - height -
                                        baseline), boxColor, thickness=cv2.FILLED)
            cv2.putText(
                img, label + " {:.2f}".format(confs[0]), (x, y - 5), font, 1, (0, 0, 0), 1)
        cv2.imshow("GeekReboot Face Mask Detection", img)


def fmProcess(frame, model, classes, output_layers):
    height, width, channels = frame.shape
    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, class_ids, classes, frame)


def sd(image):
    global processedImg
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()
    # print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    ind = []
    for i in range(0, len(classIDs)):
        if(classIDs[i] == 0):
            ind.append(i)
    a = []
    b = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)

    distance = []
    nsd = []
    for i in range(0, len(a)-1):
        for k in range(1, len(a)):
            if(k == i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d <= 100):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                # print(nsd)
    color = (0, 0, 255)
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    color = (0, 255, 0)
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'OK'
                cv2.putText(image, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    processedImg = image.copy()


if __name__ == '__main__':
    create = None
    frameno = 0

    filename = "in_videos/incrwd.mp4"
    yolo = "yolo-coco/"
    opname = "out_videos/output_of_" + filename.split('/')[1][:-4] + '.mp4'
    cap = cv2.VideoCapture(0)
    time1 = time.time()
    Setup(yolo)
    model, classes, output_layers = loadFMDYOLO()

    while(True):
        starttime = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=480)
        frameno += 1

        if(frameno % 2 == 0 or frameno == 1):
            sd(current_img)
            fmProcess(processedImg, model, classes, output_layers)
            Frame = processedImg
            #uncomment lines 221 to 224 to do the same processing for an external video file

            #if create is None:

            #    fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #    create = cv2.VideoWriter(opname, fourcc, 30, (Frame.shape[1], Frame.shape[0]), True)
            #create.write(Frame)

        stoptime = time.time()
        print("Video is Getting Processed at {:.4f} seconds per frame".format(
            (stoptime-starttime)))

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
