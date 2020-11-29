import cv2
import numpy as np
import utlis

WebCamFeed = False
ImagePath = "2.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
ImageHeight = 640
ImageWidth = 480

utlis.initializeTrackbars()
count = 0

while True:
    if WebCamFeed:success, img = cap.read()
    else:img = cv2.imread(ImagePath)    
    img = cv2.resize(img, (ImageWidth, ImageHeight)) # RESIZE IMAGE
    blank = np.zeros((ImageHeight,ImageWidth, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
 
    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
   




    imageArray = ([img,blank,blank,blank],
                      [blank,blank, blank,blank])
    
    lables = [["img","gray","blur","threshhold"],
              ["dilate","erode","ContourImage","blank"]]


    stackedImage = utlis.stackImages(imageArray,0.75,lables)
    cv2.imshow("Result",stackedImage)        
    