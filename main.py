# USAGE
# python main.py -i path/to/video.mp4 -o path/to/output.mp4

# import the necessary packages
from imutils.video import FPS
import numpy as np
import cvlib as cv
import argparse
import imutils
import pickle
import time
import cv2
import os
import time


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to video for processing")
ap.add_argument("-o", "--output",
	help="path to video for processing")
ap.add_argument("-m", "--model", default="models/openface_nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default="models/recognizer",
	help="path to model trained to recognize faces")
ap.add_argument("-e", "--label_encoder", default="models/label_encoder",
	help="path to label encoder")
args = vars(ap.parse_args())

if args["output"]:
	os.makedirs(os.path.dirname(args["output"]), exist_ok=True)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
encoder = pickle.loads(open(args["label_encoder"], "rb").read())

# initialize the video stream
print("[INFO] starting video stream...")
video = cv2.VideoCapture(args["input"])

if args["output"]:
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	out_fps = video.get(5)
	out_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	out_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out = cv2.VideoWriter(args["output"], fourcc, out_fps, (out_width,out_height))

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:

	# grab the frame from the threaded video stream
	ret, frame = video.read()

	if not ret:
		print("End of video")
		break

	results, confidences = cv.detect_face(frame) 

	for bounds in results:

		(startX, startY, endX, endY) = bounds

		# extract the face ROI and grab the ROI dimensions
		face = frame[startY:endY, startX:endX]

		try:
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			net.setInput(faceBlob)
			vec = net.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = encoder.classes_[j]

			if name == "putin":
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0), -1)
		except:
			continue

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)

	if args["output"]:
		# write frame to disk
		out.write(frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
if args["output"]:
	out.release()
video.release()
cv2.destroyAllWindows()