import tensorflow as tf


class DataGenerator():
    """Abstract class to describe a data generation object

    In general a data generator is responsible to provide data for training, validation or test.
    For this porpuse, a DataGenerator object needs to contains a a tf.data.Dataset object as dataset.

    # Args
        dataset: tf.data.Dataset. Dataset object
        labels: dict[int, str]. Labels for the dataset where keys are the label ids and the values are the label names

    """
    _dataset = None
    _labels = None

    def __init__(self, name, dataset, labels, padded_batch_shape=None, padded_values=None):
        self.name = name
        if not issubclass(dataset.__class__, tf.data.Dataset):
            raise Exception(
                "Return value of reader \"{}\" is not subclass of tf.data.Dataset!".format(name))

        self._dataset = dataset
        self._labels = labels
        self._padded_batch_shape = padded_batch_shape
        self._padded_values = padded_values

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self):
        return self._labels

    @property
    def padded_batch_shape(self):
        return self._padded_batch_shape

    @property
    def padded_values(self):
        return self._padded_values
