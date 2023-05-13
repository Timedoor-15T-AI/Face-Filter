import cv2
import itertools
import mediapipe as mp
import numpy as np

class FaceMesh():
    def __init__(self):
        self.mpfaceDetection = mp.solutions.face_detection # for detecting the face
        self.face_detection = self.mpfaceDetection.FaceDetection( # initialize the face detection model
            model_selection=0, min_detection_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils # for drawing the mesh
        self.mpFaceMesh = mp.solutions.face_mesh # for detecting the mesh
        self.faceMeshImages = self.mpFaceMesh.FaceMesh( # initialize the face mesh model
            static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5
        )
        self.faceMeshVideos = self.mpFaceMesh.FaceMesh( # initialize the face mesh model
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.3
        )
        self.mpDrawStyles = mp.solutions.drawing_styles # for drawing the mesh

    def detectFacialLandmarks(self, image, face_mesh):
        # array of facial landmarks (x, y, z)
        results = face_mesh.process(image[:, :, ::-1]) # process the image
        output_image = image[:, :, ::-1].copy() # copy the image

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks( # draw the mesh
                    image=output_image,
                    landmark_list = face_landmarks,
                    connections = self.mpFaceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = self.mpDrawStyles.get_default_face_mesh_tesselation_style()
                )

                self.mpDraw.draw_landmarks( # draw the facial landmarks
                    image=output_image,
                    landmark_list = face_landmarks,
                    connections = self.mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = self.mpDrawStyles.get_default_face_mesh_contours_style()
                )

        return np.ascontiguousarray(output_image[:, :, ::-1], dtype=np.uint8), results # return the image

    def isOpen(self, image, face_mesh_results, face_part, threshold=5):
        image_height, image_width, _ = image.shape # get the image width and height
        output_image = image.copy() # copy the image
        status = {}

        if face_part == 'MOUTH':
            INDEXES = self.mpFaceMesh.FACEMESH_LIPS # get the indexes of the mouth
        elif face_part == 'LEFT EYE':
            INDEXES = self.mpFaceMesh.FACEMESH_LEFT_EYE # get the indexes of the left eye
        elif face_part == 'RIGHT EYE':
            INDEXES = self.mpFaceMesh.FACEMESH_RIGHT_EYE # get the indexes of the right eye
        else :
            return
        
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            _, height, _ = self.getSize(image, face_landmarks, INDEXES) # get the height
            _, face_height, _ = self.getSize(image, face_landmarks, self.mpFaceMesh.FACEMESH_FACE_OVAL) # get the face height

            if ( height/face_height ) * 100 > threshold:
                status[face_no] = 'OPEN'
                color = (0, 255, 0)
            else :
                status[face_no] = 'CLOSED'
                color = (0, 0, 255)
            
            cv2.putText(output_image, f'FACE {face_no + 1} { face_part } { status[face_no] }.', 
                        (10, image_height - 30), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)
        
        return output_image, status # return the image and status
    
    def getSize(self, image, face_landmarks, INDEXES):
        image_height, image_width, _ = image.shape # get the image width and height
        INDEXES_LIST = list(itertools.chain(*INDEXES))
        landmarks = []

        for INDEX in INDEXES_LIST:
            landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width), # get the landmarks
                              int(face_landmarks.landmark[INDEX].y * image_height)]) # get the landmarks
            _, _, width, height = cv2.boundingRect(np.array(landmarks)) # get the width and height

        landmarks = np.array(landmarks) # convert to numpy array
        return width, height, landmarks # return the width, height, and landmarks
    
    def masking(self, image, filter_img, face_landmarks, face_part, INDEXES):
        annotated_image = image.copy() # copy the image

        try:
            filter_image_height, filter_image_width, _ = filter_img.shape # get the filter image width and height
            _, face_part_height, landmarks = self.getSize(image, face_landmarks, INDEXES) # get the face part height
            required_height = int(face_part_height * 2.5) # get the required height
            resized_filter_img = cv2.resize(filter_img,
                                            (int(filter_image_width * (required_height / filter_image_height)), 
                                            required_height)) # resize the filter image
            filter_image_height, filter_image_width, _ = resized_filter_img.shape # get the resized filter image width and height
            _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY_INV) # get the filter image mask
            center = landmarks.mean(axis=0).astype("int") # get the center of the face part

            if face_part == 'MOUTH':
              location = (int(center[0] - filter_image_width / 3), int(center[1]))
            else:
              location = (int(center[0] - filter_image_width / 2), int(center[1] - filter_image_height / 2))

            ROI = image[location[1]: location[1] + filter_image_height, location[0]: location[0] + filter_image_width] # get the ROI
            resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask) # get the resultant image
            resultant_image = cv2.add(resultant_image, resized_filter_img) # add the resultant image and the resized filter image

            annotated_image[location[1]: location[1] + filter_image_height, location[0]: location[0] + filter_image_width] = resultant_image # replace the ROI with the resultant image

        except Exception as e:
            pass
        
        return annotated_image # return the annotated image