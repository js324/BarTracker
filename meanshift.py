import cv2
import numpy as numpy

video = cv2.VideoCapture("video.mp4")

_, first_frame = video.read() #getting the first frame
fromCenter = False
r = cv2.selectROI(first_frame, fromCenter) 
roi = first_frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #selecting your own ROI
#If it wasn't manual, would have to do some sort of image recognizing of the barbell (?)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #hsv format (hue, saturation value), using only hue to find object
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180]) #no mask, size 180, hue goes 0-179, 180 isn't included in range
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) #normalizing to 255

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)

while True:
	_, frame = video.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1) #getting mask
	#mean-shift algorithm: finds area of highest concetration of white in mask

	_, track_window = cv2.meanShift(mask, (int(r[0]), int(r[1]), int(r[2]), int(r[3])), term_criteria) 
	x, y, w, h = track_window #coords of where the rectangle goes 
	cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

	cv2.imshow("Frame", frame) #just showing the video by getting frame by frame
	cv2.imshow("Mask", mask)
	cv2.imshow("Image", roi)
	key = cv2.waitKey(60)
	if key == 27: #if esc key is pressed, break
		break


video.release() #ending procedure
cv2.destroyAllWindows()