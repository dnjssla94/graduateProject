from tracking import Tracker, Trackable
import cv2
import numpy as np
import time

frame_size = 416    # 왜: YOLO가 처리하는 이미지가 320|416|608 이기 때문
frame_count = 0
min_confidence = 0.5
min_directions = 10

height = 0
width = 0

count_limit = 0
up_count = 0
down_count = 0
direction = ''
findObjectName = 'car'
findObjectNumber = 0

trackers = []
trackables = {}

file_name = '/Users/shinwonkim/Desktop/ai_cv/video/lane2.mp4'
output_name = '/Users/shinwonkim/Desktop/ai_cv/video/lane2__.avi'

classes = []
# .strip() : 문자열 양 끝의 공백, \n을 제거해준다.
with open("./model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

for i, name in enumerate(classes):   
    print("{}. {}".format(i, name))
    if findObjectName ==name:
        findObjectNumber = i

# Load Yolo.
net = cv2.dnn.readNet("./model/yolov3.weights", "./model/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize Tracker 
tracker = Tracker()    

writer = None   

def writeFrame(img):    #

    global writer   # 
    if writer is None and output_name is not None:  
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (img.shape[1], img.shape[0]), True) 

    if writer is not None:  
        writer.write(img) 


vs = cv2.VideoCapture(file_name)   
# loop over the frames from the video stream
while True:
    ret, frame = vs.read()  
    
    if frame is None:   
        print('### No more frame ###')
        break   
    # Start time capture
    start_time = time.time()   
    frame_count += 1     

    (height, width) = frame.shape[:2]   
    count_limit = height // 2   

    cv2.line(frame, (0, count_limit), (width, count_limit), (0, 255, 255), 2)
    
    # construct a blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (frame_size, frame_size), (0, 0, 0), True, crop=False)
    net.setInput(blob)  
    outs = net.forward(output_layers)   
    rects = []  

    confidences = []   
    boxes = []  
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)    # 몇번째가 제일 높은지.
            confidence = scores[class_id]   # 그번째의 신뢰도.
            # Filter only 'findObjectName' (사람이 0번째)
            if class_id == findObjectNumber and confidence > min_confidence:

                # Object detected
                # detection[0]: 물체 중앙 x좌표의 이미지 내 상대적 위치
                # detection[1]: 물체 중앙 y좌표의 이미지 내 상대적 위치
                # detection[2]: 물체 너비. 이미지 대비 상대적 크기로 표현
                # detection[3]: 물체 높이. 이미지 대비 상대적 크기로 표현
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates (좌측 상단 좌표 구함.)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h]) 
                
                confidences.append(float(confidence))   

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    for i in range(len(boxes)): 
        if i in indexes:
            x, y, w, h = boxes[i]
            rects.append([x, y, x+w, y+h])  
            label = '{:,.2%}'.format(confidences[i])    
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Use Tracker => 다양한 방법이 있지만 Centroid tracking algorithm을 활용할것.
    objects = tracker.update(rects)

    print("trackables: ", trackables)
    # loop over the trackable objects
    for (objectID, centroid) in objects.items():
        # check if a trackable object exists with the object ID
        trackable = trackables.get(objectID, None)
        print("objectID: ", objectID, "centroid: ", centroid)

        if trackable is None:   # 없다면 만들어 주고
                trackable = Trackable(objectID, centroid)
        else:       # trackables에 있다면 
                y = [c[1] for c in trackable.centroids] 
                variation = centroid[1] - np.mean(y) 
                trackable.centroids.append(centroid)   
                if variation < 0:   
                    direction = 1   
                else:
                    direction = 0
                trackable.directions.append(direction)
                mean_directions = int(round(np.mean(trackable.directions)))
                len_directions = len(trackable.directions)

                if (not trackable.counted) and (len_directions > min_directions):
                        if direction == 1 and centroid[1] < count_limit:
                                up_count += 1
                                trackable.counted = True
                        elif direction == 0 and centroid[1] > count_limit:
                                down_count += 1
                                trackable.counted = True

        # store the trackable object in our dictionary
        trackables[objectID] = trackable
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] + 10, centroid[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    info = [
        ("Up", up_count),
        ("Down", down_count),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    writeFrame(frame)
    
    # show the output frame
    dst = cv2.resize(frame, dsize=(840, 640), interpolation=cv2.INTER_AREA)
    cv2.imshow("Frame", dst)
    frame_time = time.time() - start_time 
    # print("Frame {} time {}".format(frame_count, frame_time))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break
        
vs.release()
writer.release()
cv2.destroyAllWindows()
