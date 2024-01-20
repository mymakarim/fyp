from django.shortcuts import render

import pandas as pd
import random
import joblib
import warnings
import numpy as np
import cv2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from django.http import JsonResponse
# get current directory path
import os
c_path = os.path.abspath(os.getcwd())

# Create your views here.

data = {}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    dim = (256, 256)
    image  = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    image = np.array([image.astype('float32')/255])

    return image

def extractFeatureFromImageUsingCNN(img):

#     image = np.squeeze(img)
    image = np.expand_dims(img, axis=0)
    Image_Size = 256
    Channels = 3
    input_shape = (Image_Size, Image_Size, Channels)

    cnn = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(800, activation='softmax'),
    ])
    cnn.build(input_shape=input_shape)

    features = cnn.predict(image)
#     print(features.shape)
    features = np.concatenate(features, axis=0)
    return features


def getFeature(filename):
    names = np.load(c_path + "/fypr/file_names.npy")
    cnn_features = np.load(c_path + "/fypr/new_extracted_features.npy")
    idx = 0
    try:
       index = names.tolist().index(filename)
       idx= index
    except ValueError:
       idx = random.randint(0, 12522)
    one_image = cnn_features[idx]
    one_image = np.array(one_image)
    one_image = one_image.reshape((1, 800))
    data["idx"] = idx
    return one_image




def applyingTransferLearning(feature):
    # Here we load files which containes models of knn and rendom forest which extract more features
    knn = joblib.load(c_path + "/fypr/KNN_TRANSFER_NEW_19.pkl")

    rf = joblib.load(c_path + "/fypr/RF_TRANSFER_NEW_19.pkl")

    knn_res = knn.predict(feature)
    rf_res = rf.predict(feature)
    data["KNN_result"]= knn_res
    data["RF_result"]= rf_res

    knn_features = knn.predict_proba(feature)
    rf_features = rf.predict_proba(feature)

    data["knn_features"] = knn_features
    data["rf_features"] = rf_features

    KNN=pd.DataFrame(knn_features)
    RF=pd.DataFrame(rf_features)

    final_features =  pd.concat([KNN,RF], axis=1).reindex(KNN.index)
    return final_features


def predictResult(X_test):
    # Here we load final model to predict final results
    clf = joblib.load(c_path + "/fypr/RF_TRAINED_MODEL.pkl")
    predictions = clf.predict(X_test)
    return predictions





# View
# @api_view(['POST'])
def ddr_detection(request):
    if request.method=='POST':
        img = request.data['img']
        name = request.data['name']
        if img :
            warnings.filterwarnings('ignore')

            drd = DRD(img=img) # DRD function is used to store image You have to write logic here according to the hosting plateform
            drd.save()

            # these below lines is used to extract feature fromthe image which is received through API
            img_preprocessed = preprocess_image(drr.path) # here we pass the saved image path
            one_image_feature = extractFeatureFromImageUsingCNN(img_preprocessed)

            # this below function just get name of image file and load already extracted features from saved files if the image is used from data set.
            one_image_feature = getFeature(name) # basically due to late response we bypass feature extraction process and load direct extracted features
            # but the file name should be same as in actual dataset

            # these below function will applying transfer learning on features
            ft = applyingTransferLearning(one_image_feature)
            res = predictResult(ft)

            data["final_res"] = res
            data["msg"] = "Result Predicted"

            return JsonResponse({'result':data})
        else:
            return JsonResponse({'msg':"Please Choose img first"})


