import numpy as np
from Wavelet_Transform import *
import cv2
import base64
import joblib
import json

path_to_original_dataset = "./test_images"
path_to_cropped_dataset = "./test_images_cropped/"

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image(image_path, image_base64_data):
    
    face_cascade = cv2.CascadeClassifier('./Server/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./Server/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        image = cv2.imread(image_path)
    else:
        image = get_cv2_image_from_base64_string(image_base64_data)
    
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cropped_faces.append(roi_image)
    return cropped_faces

def get_model_ready():
    global __class_name_to_number
    global __class_number_to_name

    path = "./Server/artifacts/class_dictionary.json"
    with open(path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    
    global __model
    if __model is None:
        path = "./Server/artifacts/saved_model.pkl"
        with open(path, 'rb') as f:
            __model = joblib.load(f)
    
    print("Done loading model...")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def classify_image(image_base64_data, image_path=None):

    imgs = get_cropped_image(image_path, image_base64_data)

    result = []
    X = []
    for img in imgs:
        scalled_raw_image = cv2.resize(img, (32, 32))
        image_har = w2d(img, 'db1', 5)
        scalled_image_har = cv2.resize(image_har, (32, 32))

        combined_img = np.vstack((scalled_raw_image.reshape(32*32*3,1),scalled_image_har.reshape(32*32,1)))
        X.append(combined_img)

        X = np.array(X).reshape(len(X), len(X[0])).astype(float)
        result.append({
            'class' : class_number_to_name(__model.predict(X)[0]),
            'class_probablity' : np.around(__model.predict_proba(X)*100,2).tolist()[0],
            'class_dictionary' : __class_name_to_number
        })
    
    return result

if __name__ == "__main__":
    get_model_ready()

    path = "./test_images/9.jpg"
    y = classify_image(None, path)

    print(y)