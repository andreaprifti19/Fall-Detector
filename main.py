import cv2
import numpy as np
import mediapipe as mp
import datetime
import time 
import pywhatkit


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
now = datetime.datetime.now()

# Video Capture
capture = cv2.VideoCapture(0)

# Grayscale for haar cascade
# gray = cv2.cvtColor(capture,cv2.COLOR_BGR2GRAY)

fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)

#Fall detected or not
fall = False
StumbleDetected = False
FallDetected = False

#Temporary
nose= 0
nosecheck = 0
noseint = 0
movement = False

#Dictionary declaration
nosedic={}
righthipdic={}
lefthipdic={}

# Keeps track of what frame we're on
frameCount = 0

#Checking of frames that fall may be detected
check = False

# Create haar cascade
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

prevTime = 0

#Pose Tracking
with mp_pose.Pose(
        min_detection_confidence=0.5, #Change if not detecting correctly
        min_tracking_confidence=0.5) as pose:
    while (1):
        # Return Value and the current frame
        ret, frame = capture.read()

        # Recoloring of capture
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Make Pose detection
        results = pose.process(frame)

        # Color back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



        # Get Landmarks
        try:

            landmarks = results.pose_landmarks.landmark
            # Defining Landmarks of body parts in Y position (vertical)
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            #Convert nose position from list index to integer
            noseint = nose[0]

            #Dictionary for body parts, containing y position and number of frame
            noselog={frameCount:nose}
            nosedic.update(noselog)

            righthiplog={frameCount:right_hip}
            righthipdic.update(righthiplog)

            lefthiplog={frameCount:left_hip}
            lefthipdic.update(lefthiplog)

            #Every 2 frames check for fall - Change to a higher/lower substractor of framecount if not working
            if frameCount>5:
                if nose>righthipdic.get(frameCount-10) and nose>lefthipdic.get(frameCount-10):
                    if fall == True:
                        pass
                    else:
                        fall = True
                        check = frameCount
                        nosecheck = nosedic.get(frameCount-10)
                        righthipcheck = righthipdic.get(frameCount-10)
                        lefthipcheck = lefthipdic.get(frameCount - 10)
                elif noseint-0.15<nosecheck:
                    fall = False

        except:
            pass

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        #If a short fall is detected give warning
        if check != 0 and frameCount== check + 10  and fall == True:
            StumbleDetected = True
            pywhatkit.sendwhatmsg_instantly('+491789295388', 'A stumble has been detected, inspection is recommended.',5 ,True, 2)


        #If person has fallen and is staying down for more than 100 frames(~10 seconds), a fall has been detected
        if check != 0 and frameCount == check + 80 and fall == True:
            FallDetected = True
            pywhatkit.sendwhatmsg_instantly('+491789295388', 'A fall has been detected and urgent inspection is recommended. In case of emergency contact: 112',5 ,True, 2)

            print("Sending")

        #If person who fell has gotten back on his feet, the fall status is removed
        if FallDetected:
            if fall==False:
                FallDetected=False
                pywhatkit.sendwhatmsg_instantly('+491789295388', 'Person seems to be standing up now, inspection is still recommended.',5 ,True, 2)


        #  Check if a current frame actually exist
        if not ret:
            break

        frameCount += 1

        # Resize the frame
        resizedFrame = cv2.resize(frame, (1280, 720), fx=0.50, fy=0.50)

        # Get the foreground mask
        fgmask = fgbg.apply(resizedFrame)

        # Count all the non zero pixels within the mask
        count = np.count_nonzero(fgmask)

        # Determines how many pixels do you want to detect to be considered "movement"
        if frameCount > 1 and count > 2500:
            movement = True
            cv2.putText(resizedFrame, 'Movement Detected: ', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
        else:
            movement = False

        gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        #Face Detection
        # Draw rectangle over face
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(resizedFrame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            #cv2.putText(resizedFrame, 'Face Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if StumbleDetected:
            cv2.putText(resizedFrame, 'Stumble Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if FallDetected:
            cv2.putText(resizedFrame, 'Fall Detected', (50, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #cv2.putText(resizedFrame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow("Camera", resizedFrame)

        #Checks for each frame
        print('\n')
        print('Frame: %d, Pixel Count: %d' % (frameCount, count))
        print(fall , "Fall checking")
        print(check, "Check Frame")
        print(StumbleDetected , "Stumble")
        print(FallDetected , "Fall")
        print(nose, "Nose")
        print(nosecheck, "NoseCheck")
        print(movement, "Movement")


        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

capture.release()
cv2.destroyAllWindows()