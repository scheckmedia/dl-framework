class Evaluator():
    def evaluate(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class ObjectDetectionEvaluator(Evaluator):
    __labels = None

    def add_batch(self, image_shape, pred_boxes, pred_classes, pred_scores,
                  gt_boxes, gt_labels, gt_ids, gt_areas=None):
        raise NotImplementedError()

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, new_labels):
        self.__labels = new_labels
