import cv2
import cvlib as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
    help="path to input video")

args = vars(ap.parse_args())

video_capture = cv2.VideoCapture(args['source'])

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    faces, confidences = cv.detect_face(frame) 

    # Draw a rectangle around the faces
    for (sx, sy, ex, ey) in faces:
        cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()