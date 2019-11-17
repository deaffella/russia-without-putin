# USAGE
# python main.py -i path/to/video.mp4 

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import cvlib as cv
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to video for processing")
ap.add_argument("-l", "--frames_limit", default=5,
	help="every n frame will be processed")
ap.add_argument("-m", "--model", default="models/openface_nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default="models/recognizer",
	help="path to model trained to recognize faces")
ap.add_argument("-e", "--label_encoder", default="models/label_encoder",
	help="path to label encoder")
args = vars(ap.parse_args())

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
encoder = pickle.loads(open(args["label_encoder"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"])
print(vs.get(cv2.CAP_PROP_FPS))

# start the FPS throughput estimator
fps = FPS().start()

frameCount = 0
sx = 0
sy = 0
ex = 0
ey = 0

# loop over frames from the video file stream
while True:

	# grab the frame from the threaded video stream
	ret, frame = vs.read()
	
	frameCount += 1

	# as_ratio = frame.shape[0] / frame.shape[1]
	# height = int(600 * as_ratio)
	# frame = cv2.resize(frame, (600, height))

	found = False

	if frameCount % int(args["frames_limit"]) == 0:
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		# frame = cv2.resize(frame, width=600)
		w = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
		h = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
					found = True
					sx = startX
					sy = startY
					ex = endX
					ey = endY
			except:
				continue

		if(found == False):
			sx = 0
			sy = 0
			ex = 0
			ey = 0

	cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 0, 0), -1)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()