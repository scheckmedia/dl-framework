import unittest
import numpy as np
import tensorflow as tf
import dlf.core
from dlf.evaluators.coco_object_detection import CocoObjectDectectionEvaluator


class TestCocoObjectDetectionEvaluator(unittest.TestCase):
    image_shape = (1024, 1024, 3)

    def setUp(self):
        self.evaluator = CocoObjectDectectionEvaluator(per_class=True)

    def test_evaluation(self):
        bboxes = np.array(
            [
                [[0.1, 0.1, 0.3, 0.3]],
                [[0.25, 0.25, 0.3, 0.3]],
                [[0.75, 0.75, 0.8, 0.8]]
            ]
        )
        labels = np.array([[1], [1], [2]])
        scores = np.array([[1.0], [1.0], [1.0]])
        ids = np.array([[0], [1], [2]])

        self.evaluator.add_batch(self.image_shape,
                                 bboxes, labels, scores,
                                 bboxes, labels, ids, None)
        result = self.evaluator.evaluate()

        self.assertEqual(result['MSCOCO_Precision/mAP'], 1.0)
        self.assertEqual(result['MSCOCO_Recall/AR@1'], 1.0)
