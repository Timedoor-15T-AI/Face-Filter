import cv2
import mediapipe

cap = cv2.VideoCapture(0) # 0 for webcam
mp_face_mesh = mediapipe.solutions.face_mesh # face mesh model
faceMesh = mp_face_mesh.FaceMesh(static_image_mode=True) # static_image_mode=True for image processing

# Function to plot the landmarks
def plot_landmark(frame, facial_area_obj):
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
        relative_source = (int(frame.shape[1] * source.x), int(frame.shape[0] * source.y))
        relative_target = (int(frame.shape[1] * target.x), int(frame.shape[0] * target.y))
        cv2.line(frame, relative_source, relative_target, (0, 255, 0), 2)

while True:
    ret, frame = cap.read() # ret is True if the frame is read correctly

    results = faceMesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # convert to RGB
    landmarks = results.multi_face_landmarks[0] # get the landmarks of the first face
    plot_landmark(frame, mp_face_mesh.FACEMESH_CONTOURS) # plot the landmarks

    cv2.imshow("Frame", frame) # show the frame
    if cv2.waitKey(1) == ord('q'): # press q to quit
        break 
    
cap.release() # release the capture
cv2.destroyAllWindows() # destroy all windows