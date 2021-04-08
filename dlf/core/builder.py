"""This module contains all functions to build the corresponding objects based
on config values.Every framework related object (model, optimizer a.s.o) shall
be generated using one of the implemented builder functions. Missing builder
functions should be implemented in this file.
"""

import tensorflow as tf
from inspect import getmodule

import dlf
from dlf.core.registry import FRAMEWORK_CALLBACKS
from dlf.core.registry import FRAMEWORK_EVALUATORS
from dlf.core.registry import FRAMEWORK_LOSSES
from dlf.core.registry import FRAMEWORK_METRICS
from dlf.core.registry import FRAMEWORK_PREPROCESSING_METHODS
from dlf.core.registry import FRAMEWORK_MODELS
from dlf.core.registry import FRAMEWORK_DATA_GENERATORS
from dlf.core.model import ModelWrapper
from dlf.core.preprocessing import PreprocessingExecutor


def build_model_wrapper(name, kwargs):
    """Builds and initializes a model based on the name parameter with kwargs
    arguments.

    Valid names are all in Keras available [Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    and all models in module dlf/models (if registered)

    # Args
        name: str. Name of the model to initialize (Keras model or a framework model)
        kwargs: dict. Required argument to initialize a model

    # Raises
        FileNotFoundError: If no model exists for a given name
        ValueError: If the registered model is not subclass of ModelWrapper or the preprocessing function is not callable
        ValueError: If something went wrong during the initialization with the kwargs

    # Returns
        A ModelWrapper object containing a the specified model, training and validation step
    """
    model = None
    preprocess_function = None

    if name in FRAMEWORK_MODELS.keys():
        model = FRAMEWORK_MODELS[name](**kwargs)
    elif name in dir(tf.keras.applications):
        fnc = getattr(tf.keras.applications, name)
        model = fnc(**kwargs)
        preprocess_function = None
        try:
            preprocess_function = getattr(
                getmodule(fnc), 'preprocess_input')
        except:
            pass

        model = ModelWrapper(model, preprocess_function)
    else:
        raise FileNotFoundError("Model \"{}\" not found!".format(name))

    try:
        if not issubclass(model.__class__, ModelWrapper) or not callable(model.preprocessing):
            raise ValueError(
                "initialized model \"{}\" returns invalid values".format(name))
        return model
    except Exception as ex:
        raise ValueError("build model \"{}\":  {}".format(name, ex))


def build_metric(name, kwargs):
    """Builds and initializes a metric based on a given name for this metric
    and with given arguments.

    Valid names are all in Keras available [Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
    and all metrics in module dlf/metrics (if registered)

    # Args
        name (str): Name of the registered metric
        kwargs (dict): Arguments for the metric to initialize

    # Raises
        ValueError: If the metric is not valid or not registered

    # Returns
        A metric with tf.keras.metrics.Metric as base class
    """
    if name in dir(tf.keras.metrics):
        return getattr(tf.keras.metrics, name)(**kwargs)
    if name in FRAMEWORK_METRICS.keys():
        return FRAMEWORK_METRICS[name](**kwargs)
    raise ValueError(
        "Invalid value for metric, check tf.keras.metrics docs for valid values")


def build_preprocessing_exectutor(pipeline):
    """Builds and initializes a PreprocessingExecutor which contains
    a list of preprocessing methods.

    Valid names are all registered methods in dlf/preprocessing (if registered)

    # Args
        pipeline: dict[str, dict[str, Any]. dictionary containing the name of a method as key and keyword arguments as values

    # Raises
        ValueError: If the requested preprocessing method not exists

    # Returns
        A dlf.core.preprocessing.PreprocessingExecutor objects
    """
    preprocessing = []
    if not isinstance(pipeline, dict):
        return preprocessing

    valid = set(FRAMEWORK_PREPROCESSING_METHODS.keys()) & pipeline.keys()
    for k in valid:
        try:
            params = pipeline[k]
            if params is not None:
                method = FRAMEWORK_PREPROCESSING_METHODS[k](**params)
            else:
                method = FRAMEWORK_PREPROCESSING_METHODS[k]()
            preprocessing += [method]
        except Exception as ex:
            raise ValueError(
                "exception for augmentation method {}\n reason: {}".format(
                    k, ex)
            )
    return PreprocessingExecutor(preprocessing)


def build_data_generator(name, args):
    """Builds and initializes a data generator.

    Valid names are all registered data generators which are located at dlf/data_generators

    # Args
        name: str. Name of the registered data generator
        args: dict[str, dict[str, Any]]. Keyword arguments for the data generator

    # Raises
        FileNotFoundError: If there is no data reader for the given name
        ValueError: If the initialization of the data reader went wrong

    # Returns
        A tf.data.Dataset object
    """

    if name not in FRAMEWORK_DATA_GENERATORS.keys():
        raise FileNotFoundError("No data reader \"%s\" found!" % name)

    try:
        generator = FRAMEWORK_DATA_GENERATORS[name](**args)
        if not issubclass(generator.__class__, dlf.core.data_generator.DataGenerator):
            raise Exception(
                "Return value of reader \"{}\" is not subclass of tf.data.Dataset!".format(name))
        return generator
    except Exception as ex:
        raise ValueError("Reader \"%s\":  %s" % (name, ex))


def build_loss(name, args):
    """Builds and initializes a loss function.

    Valid names are all in Keras available [Losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
    and all losses in module dlf/losses (if registered)

    # Args
        name: str. Name of the loss function
        args: dict[str, dict[str, Any]]. Arguments to initialize the loss function

    # Raises
        ValueError: If loss function not exists

    # Returns
        A tf.keras.losses.Loss object
    """
    if name in dir(tf.keras.losses):
        return getattr(tf.keras.losses, name)(**args)
    if name in FRAMEWORK_LOSSES.keys():
        return FRAMEWORK_LOSSES[name](**args)
    raise ValueError(
        "Invalid value for loss, check tf.keras.loss docs for valid values")


def build_optimizer(optimizer, args):
    """Builds and initializes an optimizer.

    Valid names are all in Keras available
    [Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

    # Args
        optimizer: str. Name of the optimizer to initialize
        args: dict[str, dict[str, Any]]. Intialization arguments for the selected optimizier

    # Raises
        ValueError: If optimizer name does not exists in tf.keras.optimizers
        ValueError: If an invalid decay learning object is used as learning_rate

    # Returns
        An instance of tf.keras.optimizers.Optimiizer

    """
    if optimizer not in dir(tf.keras.optimizers):
        raise ValueError(
            "Invalid value for optimizer, check tf.keras.optimizers docs for valid values")

    if 'learning_rate' in args and isinstance(args['learning_rate'], dict):
        learning_rate_method, decay_args = list(
            args['learning_rate'].items())[0]

        if learning_rate_method not in dir(tf.keras.optimizers.schedules):
            raise ValueError(
                "Invalid value for learning rate, check \"tf.keras.optimizer.schedules\""
                "docs for valid decay learning strategies"
            )

        args['learning_rate'] = getattr(
            tf.keras.optimizers.schedules, learning_rate_method)(**decay_args)

    return getattr(tf.keras.optimizers, optimizer)(**args)


def build_callback(callback, args):
    """Builds and initializes a callback object.

    Valid names are all framework callbacks located at frameworks/callback

    # Args
        callback: str. Name of the registered callback
        args: dict[str, dict[str, Any]]. Arguments to initialize the callback

    # Raises
        ValueError: If callback not exists
        ValueError: If initialization of callback went wrong

    # Returns
        An instances of dlf.core.callback.Callback
    """
    if callback not in FRAMEWORK_CALLBACKS.keys():
        raise ValueError(
            "Invalid value for callback '{}', check docs for valid values".format(callback))

    try:
        return FRAMEWORK_CALLBACKS[callback](**args)
    except Exception as ex:
        raise ValueError(
            "exception for callback {}\n reason: {}".format(
                callback, ex)
        )


def build_evaluator(evaluator, args):
    """Builds and initializes an evaluator object.

    Valid names are all framework evaluators located at frameworks/evaluator

    # Args
        evaluator: str. Name of the registered evaluator
        args: dict[str, dict[str, Any]]. Arguments to initialize the evaluator

    # Raises
        ValueError: If evaluator not exists
        ValueError: If initialization of evaluator went wrong

    # Returns
        An instances of dlf.core.evaluator.evaluator
    """
    if evaluator not in FRAMEWORK_EVALUATORS.keys():
        raise ValueError(
            "Invalid value for evaluator '{}', check docs for valid values".format(evaluator))

    try:
        if args is None:
            return FRAMEWORK_EVALUATORS[evaluator]()
        return FRAMEWORK_EVALUATORS[evaluator](**args)
    except Exception as ex:
        raise ValueError(
            "exception for evaluator {}\n reason: {}".format(
                evaluator, ex)
        )
