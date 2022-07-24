import cv2
import numpy as np

# ========================== Define Document's width and height ======================
width = 480
height = 640
# ====================================================================================

vid = cv2.VideoCapture(1)
vid.set(3, 640)
vid.set(4, 480)
vid.set(10, 100)

count = 1

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgcanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgcanny, kernel, iterations=2)
    imgThresh = cv2.erode(imgDial, kernel, iterations=1)
    return imgThresh


def getcontour(img1):
    biggest = np.array([])
    maxarea = 0
    cont, hierarchy = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in cont:
        area = cv2.contourArea(cnt)
        # print(area)
        if area>5000:
            # cv2.drawContours(img2, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            # print(len(approx))
            if area > maxarea and len(approx) == 4:
                biggest = approx
                maxarea = area
    # cv2.drawContours(img2, biggest, -1, (0, 255, 0), 20)
    return biggest

def arrangeCorners(myPoints):
    # print(myPoints)
    # print(myPoints.shape)
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("newpoints",myPointsNew)
    return myPointsNew


def getWarp(img,biggest):
    biggest = arrangeCorners(biggest)
    print(biggest.shape)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgoutput = cv2.warpPerspective(img, matrix, (width, height))
    imgcropped = imgoutput[20:imgoutput.shape[0]-20,20:imgoutput.shape[1]-20]
    imgcropped = cv2.resize(imgcropped, (width, height))

    return imgcropped

while True:
    s, img = vid.read()

    img = cv2.resize(img, (640, 480))
    img2 = img.copy()
    imgThresh = preProcess(img)
    biggest = getcontour(imgThresh)
    # print(biggest)
    # imgWarp = getWarp(img, biggest)
    if biggest.size != 0:
        imgWarp = getWarp(img, biggest)
        cv2.imshow("Video", img)
        cv2.imshow("Result", imgWarp)
    else:
        cv2.imshow("Video", img)
    key = cv2.waitKey(1)
    if key == ord('s'):  # to save drawing
        cv2.imwrite("Scanned Images/Doc no_" + str(count) + ".jpg", imgWarp)
        cv2.rectangle(imgWarp,(0,250),(480,350),(255,255,204),cv2.FILLED)
        cv2.putText(imgWarp,"Scan Saved", (50,315),cv2.FONT_HERSHEY_DUPLEX,2,(255,128,0),2)
        cv2.imshow("Result", imgWarp)
        cv2.waitKey(500)
        count += 1

    elif key == ord('q'):  # to quit
        break
