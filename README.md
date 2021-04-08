# DL-Framework
The DL-Framework is an approach to generate an experimental environment that can easily be reused and extended but without the requirements to change core functionality.
The framework is implemented using Tensorflow 2.0 in combination with tf.keras.
If you add new model functionalities, like data loading, these operations should be handled by the framework and not be implemented again and again.
The same situation for the training loop.
In a segmentation task, the training loop for all kind of networks is completely identical, it's not necessary to implement these loops for each model.

For more flexibility, the framework is configured by YAML configuration files.
This allows it, flexibly, to configure experiments without dying in an argument hell of a command-line application.
Additional pre-processing methods, metrics or loss functions can easily append or replaced just by changing the parameter in an experiment configuration.

If something is missing, don't hesitate to implement it!
A model is missing? Checkout out the already implemented models and start with your implementation.
This is also the case for loss functions, data loaders, metrics or pre-processing functions.
While semantic segmentation training or classification tasks are identically in the training loop, you can also implement your loop.
The best example is the *CycleGAN* implementation, that overrides the basic training loop.

## Docs
[Documentation](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/)

## Installation
Use the following command to install the required python packages via pip.

```sh
pip install -r requirements.txt
```

To install dlf module on your system, use

```sh
pip install .
```

## Usage
### Experiment Configuration
#### experiment:
| Key           | Summary                                                            |
| ------------- | ------------------------------------------------------------------ |
| output_folder | Required. Path where the models/weights and logging data are saved |

#### model:
The model section contains required parameters to initialize an ANN model from the framework. All list of all available models and the corresponding parameters can you find in the [Documentation/Models](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/).

#### input_reader:
The input reader section is responsible to provide data during training, validation and test *(not implemented at the moment)*. You can find all data generators and the corresponding parameters in [Documentation/Data generators](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/)

| Key               | Summary                                        |
| ----------------- | ---------------------------------------------- |
| training_reader   | Optional. Provides data used during training   |
| validation_reader | Optional. Provides data used during validation |
| training_reader   | Optional. Provides data used during training   |

#### training:
This section contains all training specific parameters. These are, for instance, the number of steps to train, after every N step start the evaluation, callbacks, metrics or just the batch size used during training.

| Key             | Summary                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| metrics         | Optional. A list of metrics which are evaluated during training/validation/test. [Documentation/Metrics](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/)                 |
| callbacks       | Optional. A list of callbacks which are executed during training/validation/test.[Documentation/Callbacks](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/)               |
| num_steps       | Required. Number of steps to train                                                                                                                                                                |
| batch_size      | Optional. Number of samples per gradient update                                                                                                                                                   |
| eval_every_step | Optional. Start evaluation at every N step (step mod N == 0)                                                                                                                                      |
| save_strategy   | Optional. Provides a strategy when a model should be saved. [Documentation](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/framework/core/experiment/#savestrategy-class) |


#### Example configuration
Below is a sample configuration that is used to train a CNN for a segmentation task.
In the experiment section, we define the output folder where all results are saved.
As model the configuration specifies the [vgg_encoder_decoder](https://dst.pages.dst.etit.tu-chemnitz.de/computer-vision/dl-framework/framework/models/VggEncoderDecoder/) and the corresponding parameters.
In this case, the input is an RGB image with a resolution of 512x512 pixels.
We set up the network that we can distinguish between 7 classes.
`Note: &num_classes is YAML syntax to define a variable which we can reuse in our configuration`
With `model_weights` the network is forced to use pre-trained weights at the given path to initialize the network.
By specifying SparseCategoricalCrossentropyIgnore as loss function we override the default loss of the model.
The fact that the `vgg_encoder_decoder` model uses a softmax-layer as the last layer we pass for the argument `from_loggits` as False value.
As mentioned before, we reuse the value of the `num_classes` variable as input for the SparseCategoricalCrossentropyIgnore objective.
As the stochastic gradient descent method, this experiment uses Adam with a learning rate of 0.00001.

During training, the network receives the input from a `tf_record_segmentation_reader`.
Not only the path to the TFRecord and Labelmap file is specified also an option to remap classes.
This functionality allows it to change categories of pixels during training from e.g. class 1 to class 0.
The `preprocess_list` contains a list of data augmentation methods that are applied before the images are fed into the network. Not only single scalars can be used as variables also lists and dictionaries.
In this case, we use the remap-list also for the validation_reader.

For a better overview of the training, the experiment uses two metrics.
First the `SparseCategoricalCrossentropyIgnore` and additional the `SparseMeanIoU`.
All these metric values are logged to Tensorboard by using the callback `SegmentationLogger` but also segmentation mask examples.
The training is executed for 50.000 steps with a batch size of 4 and after every hundredth step, the model is evaluated.
The `save_strategy` ensures that the model is only stored when the value of `validation_sparse_mean_iou` improves.


```yaml
experiment:
  output_folder: /mnt/data/experiments/segmentation/SegNet_VGG_transposed
model:
  vgg_encoder_decoder:
    input_shape:
      - 512
      - 512
      - 3
    num_classes: &num_classes 7
    summary: True
    use_skip_layers: True
    model_weights: /mnt/data/experiments/segmentation/SegNet_VGG_transposed/checkpoint
    loss:
      SparseCategoricalCrossentropyIgnore:
        num_classes: *num_classes
        from_logits: False
    optimizer:
      - Adam:
          learning_rate: 0.00001

input_reader:
  training_reader:
    name: tf_record_segmentation_reader
    path: /mnt/data/datasets/wheeled_walker_100k_8fps_25switch/25k_sample_4_mask/training.tfrecord
    labelmap: &labelmap /mnt/data/datasets/wheeled_walker_100k_8fps_25switch/label_map.pbtxt
    ignore:
    remap: &mapping
        1: 0
        2: 0
        3: 0
        4: 1
        5: 1
    preprocess_list:
      h_flip:
      v_flip:
      resize:
        width: 512
        height: 512

  validation_reader:
    name: tf_record_segmentation_reader
    path: /mnt/data/datasets/omnidetector-Flat/training-mask.tfrecord
    labelmap: *labelmap
    shuffle: False
    ignore:
    remap:
    preprocess_list:
      resize:
        width: 512
        height: 512
training:
  metrics:
    SparseCategoricalCrossentropyIgnore:
      from_logits: False
      num_classes: *num_classes
    SparseMeanIoU:
      num_classes: *num_classes
  callbacks:
    SegmentationLogger:
      num_classes: *num_classes
      # num_visualizations: 200
      opacity: 0.4
  num_steps: 50000
  eval_every_step: 100
  batch_size: 4
  save_strategy:
    monitor: validation_sparse_mean_iou
    mode: max

```

### Run an experiment
To start an experiment it is just required to execute the experiment.py with the corresponding configuration file.

```sh
python experiment.py --config config/config_dst.yml
```
