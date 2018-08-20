# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform , VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a detect function
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data # [batch, num_of_classes, num_of_occurences,(score,x0,y0,x1,y1)]
    scale = torch.Tensor([width,height,width,height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >=0.6:
            pt = (detections[0,i,j,1:]*scale).numpy()#since cv2 takes numpy array
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),3)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD net
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage,loc:storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0,117/256.0,123/256.0))
# If you want to train on a recorder video uncomment the 'Object Detection on a video' and comment out 'Object Recognition with webcam'
''' 
# Object Detection on a video
reader = imageio.get_reader('epic-horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('detected_horses.mp4',fps=fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()
'''
# Object Recognition with webcam
video_capture = cv2.VideoCapture(0) # 1 if external  camera source
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(frame,net.eval(),transform)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
