import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)
#allowing the system to sleep for 3 seconds before the webcam fires up !
time.sleep(5)
count = 0
background = 0
#Capturing the background in range of 60
for i in range(60):
    ret,background = cap.read()
background = np.flip(background,axis=1)

#when the webcam is opened, we are reading the images from the webcam
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count+=1
    img = np.flip(img,axis=1)
    
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv , lower_red , upper_red)
    
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv , lower_red , upper_red)

    mask1 = mask1+mask2

#     Refining the mask corresponding to the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5,5),np.uint8),iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),iterations=2)
    mask1 = cv2.erode(mask1,np.ones((5,5),np.uint8),iterations = 1)
    mask1 = cv2.dilate(mask1,np.ones((5,5),np.uint8),iterations = 1)
    mask2 = cv2.bitwise_not(mask1)
    
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    if contours : 
        c = max(contours , key = cv2.contourArea)
        
        x , y , w , h = cv2.boundingRect(c)
        
        cv2.Rectangle(img , (x,y),(x+w,y+h),(0,25,255),2 )
        
    # Generating the final output
    #res1 = cv2.bitwise_and(background,background,mask=mask1)
    #res2 = cv2.bitwise_and(img,img,mask=mask2)
    #final_output = cv2.addWeighted(res1,1,res2,1,0)

    cv2.imshow('image',img)
    #cv2.imshow("Harry Potter",final_output)
    k=cv2.waitKey(10)
    if k==27:
        break
        
cap.release()
cv2.destroyAllWindows()

