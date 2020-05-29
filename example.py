from face_recognition.face_recognizers import OneNeighborRecognizer
from PIL import Image

IMAGES_PATH = './datasets/platanus_all_faces'
ADVERSARY_IMAGES_PATH = './datasets/lfw-a-reduced'
TEST_IMAGE_PATH = './datasets/platanus_all_faces/andres_matte/T03CDHC6U-U917GPHT5-c7212aa1a005-512.jpeg'


clf = OneNeighborRecognizer()
clf.fit(IMAGES_PATH, ADVERSARY_IMAGES_PATH)
test = [Image.open(TEST_IMAGE_PATH)]

print(clf.ids_to_class(clf.predict(test)))
