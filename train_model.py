# USAGE
# python train_model.py -d dataset

# import the necessary packages
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input directory of face images")
ap.add_argument("-m", "--model", default="models/openface_nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default="models/recognizer",
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--label_encoder", default="models/label_encoder",
	help="path to output label encoder")
args = vars(ap.parse_args())

def main():
	data = extract_data()
	create_model(data)

def extract_data():
	# load our serialized face embedding model from disk
	print("[INFO] loading face recognizer...")
	net = cv2.dnn.readNetFromTorch(args["model"])

	# grab the paths to the input images in our dataset
	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(args["dataset"]))

	# initialize our lists of extracted faces and
	# corresponding people names
	faces = []
	names = []

	# initialize the total number of faces processed
	total = 0

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# load the image, resize it to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		face = cv2.imread(imagePath)

		try:
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			net.setInput(faceBlob)
			vec = net.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			faces.append(vec.flatten())
			names.append(name)
			total += 1
		except:
			continue

	# dump the facial faces + names to disk
	print("[INFO] serializing {} encodings...".format(total))
	return {"faces": faces, "names": names}

def create_model(data):
	# encode the labels
	print("[INFO] encoding labels...")
	encoder = LabelEncoder()
	labels = encoder.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and
	# then produce the actual face recognition
	print("[INFO] creating model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["faces"], labels)

	# write the actual face recognition model to disk
	f = open(args["recognizer"], "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

	# write the label encoder to disk
	f = open(args["label_encoder"], "wb")
	f.write(pickle.dumps(encoder))
	f.close()

main()