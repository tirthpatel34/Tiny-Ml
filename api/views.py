from django.shortcuts import render

import os
import numpy as np
import tensorflow as tf
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

# 1) Load your Keras model once
H5_PATH = "/Users/tirthpatel/Downloads/OCR-MVP-main/tomato_health_best_l2.h5"
model = tf.keras.models.load_model(H5_PATH)
IMG_SIZE = 128

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

def upload_form(request):
    return render(request, "upload.html")

@csrf_exempt
def predict_leaf(request):
    if request.method=="POST" and request.FILES.get("image"):
        f = request.FILES["image"]
        tmp = default_storage.save(f.name, f)
        full = default_storage.path(tmp)
        img  = preprocess_image(full)
        pred = model.predict(img)[0][0]
        label = "Bacterial Spot" if pred>0.5 else "Healthy"
        conf  = float(pred if pred>0.5 else 1-pred)
        # cleanup default_storage.delete(tmp) if you like
        return JsonResponse({"prediction": label, "confidence": conf})
    return JsonResponse({"error":"No image uploaded"}, status=400)

