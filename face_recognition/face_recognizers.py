import numpy as np
import torch
from .face_space import FaceSpace
from face_recognition.classifiers import OneNeighborClassifier


class OneNeighborRecognizer:

    def __init__(self):
        self.fs = FaceSpace()
        self.classifier = OneNeighborClassifier()
        self.target_classes = None

    def fit(self, target_dir, adversary_dir):
        target_embeddings, target_names = self.fs.get_embeddings(target_dir)
        adversary_embeddings, adversary_names = self.fs.get_embeddings(adversary_dir)
        embeddings = torch.cat((target_embeddings, adversary_embeddings))
        names = target_names + adversary_names
        target_classes = list(set(target_names))
        self.target_classes = np.array(target_classes)
        labels = np.array([target_classes.index(name) if name in target_classes else -1 for name in names])
        self.classifier.fit(embeddings, labels)
        return self

    def predict(self, images):
        predict_embeddings = self.fs.get_embeddings_from_images(images)
        return self.classifier.predict(predict_embeddings)

    def ids_to_class(self, ids):
        return self.target_classes[ids]
