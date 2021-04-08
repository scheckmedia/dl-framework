from abc import ABC, abstractmethod


class PreprocessingJob():
    """A container that hosts images, masks and bounding boxes for preprocessing

    # Arguments
        image: tf.tensor. Image that should be preprocessed
        mask: tf.tensor, optional. Mask that should be preprocessed. Defaults to None.
        boxes: tf.tensor, optional. Bounding Boxes to preprocess. Defaults to None.
    """

    def __init__(self, image, mask=None, boxes=None):
        super().__init__()
        self.image = image
        self.mask = mask
        self.boxes = boxes


class PreprocessingExecutor():
    """Object that applies a list of preprocessing methods to images,masks and boxes

    # Arguments
        preprocessing_methods: [PreprocessingMethod]. List of preprocessing methods which should be applied
    """

    def __init__(self, preprocessing_methods):
        super().__init__()
        self.preprocessing_methods = preprocessing_methods

    def __call__(self, image, mask=None, boxes=None):
        job = PreprocessingJob(image, mask, boxes)
        for method in self.preprocessing_methods:
            job = method(job)

        return job


class PreprocessingMethod(ABC):
    """Abstract class to describe a Preprocessing Object

    In general a preprocessing method is executed for an image.
    Optional you can pass a mask or a pair image to apply the same transformation
    like it is done for the image.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, processing_job: PreprocessingJob):
        """Executed during preprocessing step

        # Args
            processing_job: PreprocessingJob. Job that should be preprocessed
        """
        pass
