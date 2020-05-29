from abc import ABC, abstractmethod
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import torch
from face_recognition.helpers.simple_metric_report import report

DEFAULT_THRESHOLD = 1
DEFAULT_DISTANCE = "euclidean"
DEFAULT_INITIAL_TRAIN_STEP = 0.5
DEFAULT_FP = 0
DEFAULT_STEP_MODIFIER = 3 / 4
ACC_STATIONARY_STOP = 10
MAX_TRAIN_ITERATIONS = 100


def get_false_positives(labels, predictions):
    conf_matrix = confusion_matrix(labels, predictions)
    wrong_not_recognized = sum((row[0] for row in conf_matrix[1:]))
    hits = accuracy_score(labels, predictions, normalize=False)
    return len(labels) - hits - wrong_not_recognized


class DistanceClassifier(ABC):
    def __init__(self, threshold=DEFAULT_THRESHOLD, distance_metric=DEFAULT_DISTANCE):
        self.threshold = threshold
        self.distance_metric = DistanceMetric.get_metric(distance_metric)
        self.embeddings = torch.Tensor([])
        self.labels = np.array([])

    def __train_threshold(
        self,
        embeddings,
        labels,
        max_fp,
        initial_step,
        step_modifier,
        max_train_iterations,
        acc_stationary_stop,
    ):
        step = initial_step
        step_sign = -1
        previous_accuracy = 0
        false_positives = max_fp + 1
        accuracy_stationary_count = 0
        iteration_count = 0
        for _ in range(max_train_iterations):
            predictions = self.predict(embeddings)
            actual_accuracy = accuracy_score(labels, predictions)
            false_positives = get_false_positives(labels, predictions)
            if previous_accuracy == actual_accuracy:
                accuracy_stationary_count += 1
            else:
                accuracy_stationary_count = 0
            if (
                accuracy_stationary_count >= acc_stationary_stop
                and false_positives <= max_fp
            ):
                break
            if false_positives > max_fp:
                if step_sign == 1:
                    step_sign *= -1
                    step = step * step_modifier
            elif actual_accuracy < previous_accuracy:
                step_sign *= -1
                step = step * step_modifier
            self.threshold += step_sign * step
            iteration_count += 1
            previous_accuracy = actual_accuracy
        return actual_accuracy, false_positives

    def fit(
        self,
        embeddings,
        labels,
        max_fp=DEFAULT_FP,
        initial_threshold=DEFAULT_THRESHOLD,
        initial_train_step=DEFAULT_INITIAL_TRAIN_STEP,
        step_modifier=DEFAULT_STEP_MODIFIER,
        max_train_iterations=MAX_TRAIN_ITERATIONS,
        acc_stationary_stop=ACC_STATIONARY_STOP,
        verbose=False
    ):
        self.threshold = initial_threshold
        first_occ_labels, first_occ_indices = np.unique(labels, return_index=True)
        first_unknown_index = np.argwhere(first_occ_labels == -1)
        first_occ_indices = np.delete(first_occ_indices, first_unknown_index)
        train_set_indices = np.setdiff1d(np.arange(len(labels)), first_occ_indices)
        self.labels = labels[first_occ_indices]
        self.embeddings = embeddings[first_occ_indices]
        final_acc, final_fp = self.__train_threshold(
            embeddings[train_set_indices],
            labels[train_set_indices],
            max_fp,
            initial_train_step,
            step_modifier,
            max_train_iterations,
            acc_stationary_stop,
        )
        if verbose:
            report(
                {
                    "accuracy": final_acc,
                    "false positives": final_fp,
                    "threshold": self.threshold,
                }
            )

    @abstractmethod
    def predict(self, test_embeddings):
        pass
