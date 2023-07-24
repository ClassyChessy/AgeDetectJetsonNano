import sys
import argparse
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, cudaFromNumpy, cudaToNumpy 
import numpy as np

net = detectNet(model="trafficcamnet", labels = "trafficcamnet_pruned_v1.0.3/labels.txt",
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                 threshold=0.6)
net.SetTrackingEnabled = True
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)
input = videoSource("/dev/video0", argv=sys.argv)
output = videoOutput("test/test_f1", argv=sys.argv)
	
# load the object detection network

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

# process frames until EOS or the user exits
def dist(x,y):
    x_val = abs(x[0] - y[0])
    y_val = abs(x[1] - y[1])
    hypo = (x_val **2 + y_val ** 2) ** (1/2)
    return hypo
def speed(y, time):
    mylist = []
    for i in range(5):    
        mylist[i] = y[i]/time[i]
    min = min(mylist)
    return min
x = 0
time = []
holder = []
mins = []
while True:
    if x == 5:
        x = 0
        mins.append(speed(holder, time))
        if min(mins) <= 8:
            mins = []
            print("This person did not speed past the stopsign.")
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay="box,labels,conf")

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        stopsign = (0,0)
        car = (0,0)
        print(detection)
        if detection.ClassID == 0 or detection.ClassID == 3:
            car = detection.Center
            carTS = detection.TrackStatus
            if carTS == -1:
                print("Car is out of view.")
                if min(mins) > 8:
                    print("Car was speeding. ")
                    mins = []
        elif detection.ClassID == 2:
            stopsign = detection.Center
        holder.insert((x%5), dist(car,stopsign))
        time.insert((x%5), net.getNetworkTime())

    # render the image
   # output.Render(img)

    # update the title bar
    #output.SetStatus(" | Network {:.0f} FPS".format( net.GetNetworkFPS()))

    # print out performance info
    print("Your FPS was: " , net.GetNetworkFPS())
  #  net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
    x += 1
