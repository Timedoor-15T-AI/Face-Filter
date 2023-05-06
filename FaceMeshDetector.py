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
        image_height, image_width, _ = image.shape # get the image height and width
        output_image = image.copy() # copy the image
        status = {}

        if face_part == 'MOUTH':
            INDEXES = self.mpFaceMesh.FACEMESH_LIPS
        elif face_part == 'LEFT EYE':
            INDEXES = self.mpFaceMesh.FACEMESH_LEFT_EYE
        elif face_part == 'RIGHT EYE':
            INDEXES = self.mpFaceMesh.FACEMESH_RIGHT_EYE
        else:
            return

        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            _, height, _ = self.getSize(image, face_landmarks, INDEXES) # get the height of the bounding box
            _, face_height, _ = self.getSize(image, face_landmarks, self.mpFaceMesh.FACEMESH_FACE_OVAL) # get the height of the face
            if (height / face_height) * 100 > threshold:
                status[face_no] = 'OPEN'
                color = (0, 255, 0)
            else:
                status[face_no] = 'CLOSED'
                color = (0, 0, 255)

            cv2.putText(output_image, f'FACE {face_no + 1} {face_part} {status[face_no]}.', (10, image_height - 30), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)

        return output_image, status
    
    def getSize(self, image, face_landmarks, INDEXES):
        image_height, image_width, _ = image.shape # get the image height and width
        INDEXES_LIST = list(itertools.chain(*INDEXES)) # convert the INDEXES to a list
        landmarks = []

        for INDEX in INDEXES_LIST:
            landmarks.append([
                int(face_landmarks.landmark[INDEX].x * image_width), # get the x coordinate
                int(face_landmarks.landmark[INDEX].y * image_height) # get the y coordinate
            ])

        _, _, width, height = cv2.boundingRect(np.array(landmarks)) # get the width and height of the bounding box
        landmarks = np.array(landmarks) # convert the landmarks to a numpy array
        return width, height, landmarks
    
    def masking(self, image, filter_img, face_landmarks, face_part, INDEXES):
        anotated_image = image.copy() # copy the image

        try :
            filter_img_height, filter_img_width, _ = filter_img.shape # get the width and height of the filter image
            _, face_part_height, landmarks = self.getSize(image, face_landmarks, INDEXES) # get the height of the face part
            required_height = int(face_part_height * 2.5) # get the required height
            resized_filter_image = cv2.resize(filter_img, (int(filter_img_width * (required_height / filter_img_height)), required_height)) # resize the filter image
            filter_img_height, filter_img_width, _ = resized_filter_image.shape # get the width and height of the resized filter image
            _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_image, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY_INV) # get the filter image mask
            center = landmarks.mean(axis=0).astype("int") # get the center of the face part

            if face_part == 'MOUTH':
                location = (int(center[0] - filter_img_width / 3 ), int(center[1])) # get the location of the filter image
            else:
                location = (int(center[0] - filter_img_width / 2 ), int(center[1] - filter_img_height / 2))
            

            ROI = image[location[1]:location[1] + filter_img_height, location[0]:location[0] + filter_img_width] # get the ROI
            resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask) # get the resultant image
            resultant_image = cv2.add(resultant_image, resized_filter_image) # add the filter image to the resultant image
            anotated_image[location[1]:location[1] + filter_img_height, location[0]:location[0] + filter_img_width] = resultant_image # replace the ROI with the resultant image
        
        except Exception as e:
            pass
        
        return anotated_image