# based on https://github.com/chainer/chainercv/blob/v0.13.1/chainercv/evaluations/eval_detection_coco.py#L16

import itertools
import os

import numpy as np

from dlf.core.evaluator import ObjectDetectionEvaluator
from dlf.core.registry import register_evaluator
from dlf.utils.helpers import RedirectOut

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    _available = True
except ImportError:
    _available = False


@register_evaluator(
    'coco_object_detection_evaluator',
    'CocoObjectDectectionEvaluator')
class CocoObjectDectectionEvaluator(ObjectDetectionEvaluator):
    """Evaluator object that evaluates object detection results based on MS COCO metrics.

    # Arguments
        per_class: bool. If true, all metrics are additionally logged for each class. Defaults to True.

    # Raises
        ValueError: If pycocotools are not installed
    """

    __coco_key_mapping = {
        'map/iou=0.50:0.95/area=all/max_dets=100': 'MSCOCO_Precision/mAP',
        'map/iou=0.50/area=all/max_dets=100': 'MSCOCO_Precision/mAP@.50IOU',
        'map/iou=0.75/area=all/max_dets=100': 'MSCOCO_Precision/mAP@.75IOU',
        'mar/iou=0.50:0.95/area=all/max_dets=1': 'MSCOCO_Recall/AR@1',
        'mar/iou=0.50:0.95/area=all/max_dets=10': 'MSCOCO_Recall/AR@10',
        'mar/iou=0.50:0.95/area=all/max_dets=100': 'MSCOCO_Recall/AR@100',
        'map/iou=0.50:0.95/area=small/max_dets=100': 'MSCOCO_Precision/mAP (small)',
        'map/iou=0.50:0.95/area=medium/max_dets=100': 'MSCOCO_Precision/mAP (medium)',
        'map/iou=0.50:0.95/area=large/max_dets=100': 'MSCOCO_Precision/mAP (large)',
        'mar/iou=0.50:0.95/area=small/max_dets=100': 'MSCOCO_Recall/AR@100 (small)',
        'mar/iou=0.50:0.95/area=medium/max_dets=100': 'MSCOCO_Recall/AR@100 (medium)',
        'mar/iou=0.50:0.95/area=large/max_dets=100': 'MSCOCO_Recall/AR@100 (large)',

        # only relevant if per_class is true
        'ap/iou=0.50:0.95/area=all/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP',
        'ap/iou=0.50/area=all/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP@.50IOU',
        'ap/iou=0.75/area=all/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP@.75IOU',
        'ar/iou=0.50:0.95/area=all/max_dets=1': 'MSCOCO_Class_{}/Recall/AR@1',
        'ar/iou=0.50:0.95/area=all/max_dets=10': 'MSCOCO_Class_{}/Recall/AR@10',
        'ar/iou=0.50:0.95/area=all/max_dets=100': 'MSCOCO_Class_{}/Recall/AR@100',
        'ap/iou=0.50:0.95/area=small/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP (small)',
        'ap/iou=0.50:0.95/area=medium/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP (medium)',
        'ap/iou=0.50:0.95/area=large/max_dets=100': 'MSCOCO_Class_{}/Precision/mAP (large)',
        'ar/iou=0.50:0.95/area=small/max_dets=100': 'MSCOCO_Class_{}/Recall/AR@100 (small)',
        'ar/iou=0.50:0.95/area=medium/max_dets=100': 'MSCOCO_Class_{}/Recall/AR@100 (medium)',
        'ar/iou=0.50:0.95/area=large/max_dets=100': 'MSCOCO_Class_{}/Recall/AR@100 (large)'
    }

    __cached_gt_annotations = {}
    __cached_ids = []

    def __init__(self, per_class=True):
        if not _available:
            raise ValueError(
                'Please install pycocotools \n'
                'pip install -e \'git+https://github.com/cocodataset/coco.git'
                '#egg=pycocotools&subdirectory=PythonAPI\'')
        self.per_class = per_class
        self.gt_coco = pycocotools.coco.COCO()
        self.pred_coco = pycocotools.coco.COCO()
        self.ids = []
        self.pred_annos = []
        self.gt_annos = []
        self.existent_labels = {}
        self.__id_counter = 0

    def add_batch(self, image_shape, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_labels, gt_ids, gt_areas):
        self.__convert_to_coco(image_shape, pred_boxes, pred_classes,
                               pred_scores, gt_boxes, gt_labels, gt_ids, gt_areas)

    def evaluate(self):
        existent_labels = sorted(self.existent_labels.keys())
        self.pred_coco.dataset['categories'] = [
            {'id': i} for i in existent_labels]
        self.gt_coco.dataset['categories'] = [{'id': i}
                                              for i in existent_labels]
        self.pred_coco.dataset['annotations'] = self.pred_annos
        self.gt_coco.dataset['annotations'] = self.gt_annos
        self.pred_coco.dataset['images'] = self.ids
        self.gt_coco.dataset['images'] = self.ids

        with RedirectOut(os.devnull) as out:
            self.pred_coco.createIndex()
            self.gt_coco.createIndex()
            self.coco_eval = pycocotools.cocoeval.COCOeval(
                self.gt_coco, self.pred_coco, 'bbox')
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()

        results = {'coco_eval': self.coco_eval}
        p = self.coco_eval.params
        common_kwargs = {
            'prec': self.coco_eval.eval['precision'],
            'rec': self.coco_eval.eval['recall'],
            'iou_threshs': p.iouThrs,
            'area_ranges': p.areaRngLbl,
            'max_detection_list': p.maxDets}

        all_kwargs = {
            'ap/iou=0.50:0.95/area=all/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'all',
                'max_detection': 100},
            'ap/iou=0.50/area=all/max_dets=100': {
                'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
                'max_detection': 100},
            'ap/iou=0.75/area=all/max_dets=100': {
                'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=all/max_dets=1': {
                'ap': False, 'iou_thresh': None, 'area_range': 'all',
                'max_detection': 1},
            'ar/iou=0.50:0.95/area=all/max_dets=10': {
                'ap': False, 'iou_thresh': None, 'area_range': 'all',
                'max_detection': 10},
            'ar/iou=0.50:0.95/area=all/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'all',
                'max_detection': 100},
            'ap/iou=0.50:0.95/area=small/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'small',
                'max_detection': 100},
            'ap/iou=0.50:0.95/area=medium/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 100},
            'ap/iou=0.50:0.95/area=large/max_dets=100': {
                'ap': True, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=small/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'small',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=medium/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'medium',
                'max_detection': 100},
            'ar/iou=0.50:0.95/area=large/max_dets=100': {
                'ap': False, 'iou_thresh': None, 'area_range': 'large',
                'max_detection': 100}
        }

        for key, kwargs in all_kwargs.items():
            kwargs.update(common_kwargs)
            metrics, mean_metric = self.__summarize(**kwargs)

            # pycocotools ignores classes that are not included in
            # either gt or prediction, but lies between 0 and
            # the maximum label id.
            # We set values for these classes to np.nan.
            results[key] = np.nan * np.ones(np.max(existent_labels) + 1)
            results[key][existent_labels] = metrics
            results['m' + key] = mean_metric

        results['existent_labels'] = existent_labels
        results = self.__convert_to_tensorboard(results)

        self.clear()

        return results

    def clear(self):
        self.ids.clear()
        self.pred_annos.clear()
        self.gt_annos.clear()
        # self.existent_labels.clear()
        self.__id_counter = 0

    def __convert_to_tensorboard(self, coco_results):
        results = {}
        for k, v in coco_results.items():
            if k not in self.__coco_key_mapping:
                continue

            nk = self.__coco_key_mapping[k]

            if self.per_class and not k.startswith('m'):
                for cid in coco_results['existent_labels']:
                    label = cid

                    if self.labels is not None and cid in self.labels:
                        label = self.labels[cid]

                    # label = self.labels[cid] if self.labels is not None and cid in self.labels else cid
                    label = nk.format(str(label))
                    results[label] = v[cid]
            else:
                results[nk] = v

        return results

    def __convert_to_coco(self, image_shape, pred_bboxes, pred_labels, pred_scores,
                          gt_bboxes, gt_labels, gt_ids, gt_areas, gt_crowdeds=None):
        pred_bboxes = iter(pred_bboxes)
        pred_labels = iter(pred_labels)
        pred_scores = iter(pred_scores)
        gt_bboxes = iter(gt_bboxes)
        gt_labels = iter(gt_labels)
        gt_ids = iter(gt_ids)
        gt_areas = (iter(gt_areas) if gt_areas is not None
                    else itertools.repeat(None))
        gt_crowdeds = (iter(gt_crowdeds) if gt_crowdeds is not None
                       else itertools.repeat(None))

        image_shape = np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        for i, (pred_bbox, pred_label, pred_score, gt_bbox, gt_label,
                gt_id, gt_area, gt_crowded) in enumerate(zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_ids, gt_areas, gt_crowdeds)):
            if gt_area is None:
                gt_area = itertools.repeat(None)
            if gt_crowded is None:
                gt_crowded = itertools.repeat(None)
            # Starting ids from 1 is important when using COCO.

            if gt_id not in self.__cached_ids:
                self.__cached_ids.append(gt_id)

            image_id = self.__cached_ids.index(gt_id) + 1

            for pred_bb, pred_lb, pred_sc in zip(pred_bbox, pred_label,
                                                 pred_score):
                self.pred_annos.append(
                    self.__create_anno(pred_bb * image_shape, pred_lb, pred_sc,
                                       img_id=image_id, anno_id=len(
                                           self.pred_annos) + 1,
                                       crw=0, ar=None))
                self.existent_labels[pred_lb] = True

            for gt_bb, gt_lb, gt_ar, gt_crw in zip(
                    gt_bbox, gt_label, gt_area, gt_crowded):
                self.gt_annos.append(
                    self.__create_anno(gt_bb * image_shape, gt_lb, None,
                                       img_id=image_id, anno_id=len(
                                           self.gt_annos) + 1,
                                       ar=gt_ar, crw=gt_crw))
                self.existent_labels[gt_lb] = True

            self.ids.append({'id': image_id})

    def __create_anno(self, bb, lb, sc, img_id, anno_id, ar=None, crw=None):
        y_min = bb[0]
        x_min = bb[1]
        y_max = bb[2]
        x_max = bb[3]
        height = y_max - y_min
        width = x_max - x_min
        if ar is None:
            # We compute dummy area to pass to pycocotools.
            # Note that area dependent scores are ignored afterwards.
            ar = height * width
        if crw is None:
            crw = False
        # Rounding is done to make the result consistent with COCO.
        anno = {
            'image_id': img_id, 'category_id': lb,
            'bbox': [np.round(x_min, 2), np.round(y_min, 2),
                     np.round(width, 2), np.round(height, 2)],
            'segmentation': [x_min, y_min, x_min, y_max,
                             x_max, y_max, x_max, y_min],
            'area': ar,
            'id': anno_id,
            'iscrowd': crw}
        if sc is not None:
            anno.update({'score': sc})
        return anno

    def __summarize(self,
                    prec, rec, iou_threshs, area_ranges,
                    max_detection_list,
                    ap=True, iou_thresh=None, area_range='all',
                    max_detection=100):
        a_idx = area_ranges.index(area_range)
        m_idx = max_detection_list.index(max_detection)
        if ap:
            val_value = prec.copy()  # (T, R, K, A, M)
            if iou_thresh is not None:
                val_value = val_value[iou_thresh == iou_threshs]
            val_value = val_value[:, :, :, a_idx, m_idx]
        else:
            val_value = rec.copy()  # (T, K, A, M)
            if iou_thresh is not None:
                val_value = val_value[iou_thresh == iou_threshs]
            val_value = val_value[:, :, a_idx, m_idx]

        val_value[val_value == -1] = np.nan
        val_value = val_value.reshape((-1, val_value.shape[-1]))
        valid_classes = np.any(np.logical_not(np.isnan(val_value)), axis=0)
        cls_val_value = np.nan * np.ones(len(valid_classes), dtype=np.float32)
        cls_val_value[valid_classes] = np.nanmean(
            val_value[:, valid_classes], axis=0)

        if not np.any(valid_classes):
            mean_val_value = np.nan
        else:
            mean_val_value = np.nanmean(cls_val_value)
        return cls_val_value, mean_val_value
