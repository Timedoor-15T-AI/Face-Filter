import cv2
import FaceMeshDetector as fmd

detector = fmd.FaceMesh() # initialize the FaceMesh class

cap = cv2.VideoCapture(0) # 0 for webcam
cap.set(3, 1280) # set width
cap.set(4, 720) # set height
left_eye = cv2.imread("./assets/eye1.png") # read the left eye image
right_eye = cv2.imread("./assets/eye2.png") # read the right eye image
smoke_animation = cv2.VideoCapture("./assets/smoke_animation.mp4") # read the smoke animation video
smoke_frame_counter = 0 # initialize the smoke frame counter

while True:
    ret, frame = cap.read() # ret is True if the frame is read correctly
    ret, smoke_frame = smoke_animation.read() # read the smoke frame
    smoke_frame_counter += 1 # increment the smoke frame counter
    if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT): # if the smoke frame counter is equal to the total number of frames in the smoke animation video
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0) # set the smoke animation video to the first frame
        smoke_frame_counter = 0 # reset the smoke frame counter
    frame = cv2.flip(frame, 1) # flip the frame horizontally
    frame_face_mesh, face_mesh_result = detector.detectFacialLandmarks(frame, detector.faceMeshVideos) # detect the facial landmarks

    cv2.imshow("Frame", frame_face_mesh) # show the frame
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release() # release the capture
cv2.destroyAllWindows() # destroy all windows