import sys
import argparse
# First thing i need to do is get my faces 
from jetson_inference import detectNet, imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, cudaFromNumpy, cudaToNumpy 
import numpy as np
font = cudaFont()
from PIL import Image
# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="test/test_f2.png", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="facedetect", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.7, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
        
	parser.print_help()
	sys.exit(0)
#
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network

net = detectNet(args.network, sys.argv, args.threshold)
net2 = imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")
# note: to hard-code the paths to load a model, the following API can be used:
#
#net = detectNet(model="facedetect", labels="labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

# process frames until EOS or the user exits

while True:
    #capture the next image
    img = input.Capture()
    if img is None: # timeout
        continue  
    python_image = Image.fromarray(cudaToNumpy(img))
    # detect objects in the image (with overlay)    
    detections = net.Detect(img , overlay=args.overlay)
    
    # print the detections
    #print("detected {:d} faces in image".format(len(detections)))

    for detection in detections:
        face_img = cudaFromNumpy(np.asarray(python_image.crop((detection.Left,detection.Top,detection.Right,detection.Bottom))))
        predictions = net2.Classify(face_img, topK = 1)
        for n, (classID, confidence) in enumerate(predictions):
            classLabel = net.GetClassLabel(classID)
            confidence *= 100.0
            print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")
            font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", x=5, y=5 + n * (font.GetSize() + 5),color=font.White, background=font.Gray40)
    # render the image
        output.Render(img)

    # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
        net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
