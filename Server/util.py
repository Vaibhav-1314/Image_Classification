import numpy as np
from Wavelet_Transform import *
import cv2
import os
import shutil
import joblib

path_to_original_dataset = "./test_images"
path_to_cropped_dataset = "./test_images_cropped/"

MODEL = None

def get_cropped_image(image_path):
    
    face_cascade = cv2.CascadeClassifier('./Server/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./Server/opencv/haarcascades/haarcascade_eye.xml')

    image = cv2.imread(image_path)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) == 2:
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            return roi_image

def get_model_ready():
    global MODEL

    if MODEL is None:
        path = "./Server/artifacts/saved_model.pkl"
        with open(path, 'rb') as f:
            MODEL = joblib.load(f)
            print("Done loading model...")

def classify_image(image_path=None):
    if os.path.exists(path_to_cropped_dataset):
        shutil.rmtree(path_to_cropped_dataset)
    os.mkdir(path_to_cropped_dataset)

    if image_path is not None:
        cropped_img = get_cropped_image(image_path)
        if cropped_img is not None:
            cropped_img_path = path_to_cropped_dataset + "0.jpg"
            cv2.imwrite(cropped_img_path, cropped_img)
        else:
            print("Error...")
            return None
        X = []
        raw_image = cv2.imread(cropped_img_path)
        if raw_image is None:
            print("Error...")
            return None
        scalled_raw_image = cv2.resize(raw_image, (32, 32))
        image_har = w2d(raw_image, 'db1', 5)
        scalled_image_har = cv2.resize(image_har, (32, 32))

        combined_img = np.vstack((scalled_raw_image.reshape(32*32*3,1),scalled_image_har.reshape(32*32,1)))
        X.append(combined_img)

        X = np.array(X).reshape(len(X), len(X[0])).astype(float)
        y = MODEL.predict_proba(X)
        result = []
        for x in y:
            if(x[0] > x[1]):
                result.append(0)
            else:
                result.append(1)    
        return result

# ------------------------------------------------>

    i = 1
    for entry in os.scandir(path_to_original_dataset):
        # print(entry.path)
        cropped_img = get_cropped_image(entry.path)
        
        if cropped_img is not None:
            cropped_img_path = path_to_cropped_dataset + str(i) + ".jpg"
            cv2.imwrite(cropped_img_path, cropped_img)
            i += 1

    X = []
    for image in os.scandir(path_to_cropped_dataset):
        raw_image = cv2.imread(image.path)

        if raw_image is None:
            continue
        
        scalled_raw_image = cv2.resize(raw_image, (32, 32))
        image_har = w2d(raw_image, 'db1', 5)
        scalled_image_har = cv2.resize(image_har, (32, 32))

        combined_img = np.vstack((scalled_raw_image.reshape(32*32*3,1),scalled_image_har.reshape(32*32,1)))
        X.append(combined_img)

    X = np.array(X).reshape(len(X), len(X[0])).astype(float)

    y = MODEL.predict_proba(X)
    result = []

    for x in y:
        if(x[0] > x[1]):
            result.append(0)
        else:
            result.append(1)
    return result

if __name__ == "__main__":
    get_model_ready()

    y = classify_image("./test_images/11.jpg")

    print(y)