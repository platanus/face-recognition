from .distance_classifier import DistanceClassifier
import numpy as np


class OneNeighborClassifier(DistanceClassifier):
    def predict(self, test_embeddings):
        predictions = []
        pairwise_distances = self.distance_metric.pairwise(
            test_embeddings, self.embeddings
        )
        for sample_distances in pairwise_distances:
            min_distance = np.amin(sample_distances)
            if min_distance < self.threshold:
                predictions.append(np.argmin(sample_distances))
            else:
                predictions.append(-1)
        return predictions
