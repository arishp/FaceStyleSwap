# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install insightface
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# img = ins_get_image('t1')
img = cv2.imread('m_1.jpg')
# plt.imshow(img[:,:,::-1])
# plt.show()

faces = app.get(img)
print(len(faces))
print(faces[0].keys())

# fig, axs = plt.subplots(1, len(faces), figsize=(12, 5))
# for i, face in enumerate(faces):
#     bbox = face['bbox']
#     bbox = [int(b) for b in bbox]
#     axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
#     axs[i].axis('off')
# plt.show()

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
kayal_img = cv2.imread('kayal_3.jpg')
# plt.imshow(kayal_img[:,:,::-1])
# plt.show()

kayal_faces = app.get(kayal_img)
kayal_face = kayal_faces[0]
res = img.copy()
# for face in faces:
#     res = swapper.get(res, face, kayal_face, paste_back=True)
res = swapper.get(res, faces[1], kayal_face, paste_back=True)
# plt.imshow(res[:,:,::-1])
# plt.show()

arish_img = cv2.imread('arish_3.jpg')
arish_faces = app.get(arish_img)
arish_face = arish_faces[0]
res = swapper.get(res, faces[0], arish_face, paste_back=True)
plt.imshow(res[:,:,::-1])
plt.show()

cv2.imwrite('res.jpg', res)
