import tensorflow as tf # type: ignore
from transformers import AutoConfig, TFAutoModelForImageClassification
from  tensorflow.keras.applications import InceptionV3, ResNet50V2, ConvNeXtBase # type: ignore
from tf_keras.losses import BinaryCrossentropy # type: ignore
from tf_keras.callbacks import Callback # type: ignore
from tf_keras.metrics import BinaryAccuracy, AUC # type: ignore
from tf_keras.optimizers import AdamW # type: ignore


def _get_cnn(name, input_shape):
  """Helper method to load tf CNN"""
  if(name == "Convnext"):
    return ConvNeXtBase(include_top = False, input_shape = input_shape)
  if(name == "Inceptionv3"):
    return InceptionV3(include_top = False, input_shape = input_shape)
  if(name == "Resnet50"):
    return ResNet50V2(include_top = False, input_shape = input_shape)
  raise ValueError(f"name: {name} is not a valid input")

def get_feature_extractor(name, input_shape):
  """Get TF CNN feature extractor  model.
  
  Args:
    name: Name of the CNN to be used. Valid inputs are 'Convnext', 'InceptionV3', 'Resnet50'.
    input_shape: Shape of the images in (w, h, c) format.
  """
  cnn = _get_cnn(name, input_shape)
  permute = permute = tf.keras.layers.Permute((3, 1, 2)) # move channels to the front
  return tf.keras.Sequential([cnn, permute])

def get_vit(hf_id, img_size, num_channels, hf_kwargs):
  """Loads ViT from the transformers library.
  
  Loads a Hugging Face transformer as a Keras model and compiles it.
  The optimizer is AdamW, the loss function is BinaryCrossentropy and the metrics
  are binary accuracy and AUC.

  Args:
    hf_id: URL or filepath to the model to be loaded.
    img_size: Width or height of image given to the model.
    num_channels: Number of channels of the given image
    hf_kwargs: Arguments to be passed to AutoConfig.from_pretrained
  """
  # patch_size = 1, problem_type = "single_label_classification", num_labels=1
  config = AutoConfig.from_pretrained(hf_id, image_size = img_size, num_channels = num_channels, **hf_kwargs)
  model = TFAutoModelForImageClassification.from_config(config)

  optimizer = AdamW(learning_rate=1e-5)
  loss = BinaryCrossentropy(from_logits = True)
  metrics = [
    BinaryAccuracy(name = "binary_accuracy"),
    AUC(from_logits=True, name = "auc")
  ]
  
  model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics,
  )
  return model

@tf.function
def call_all_vits(img, vits):
  """Calls each ViT on a given image and combines their output.

  Concatenates each predicted output from the Vision Transformers.
  Used to map the tensorflow dataset to train the meta network.

  Args:
    img: The image to be called.
    vits: A list of vits used to predict.
  """
  outputs = []
  for vit in vits:
    outputs.append(vit(img).logits)
  x = tf.keras.layers.Concatenate()(outputs)
  return x

def get_meta(num_classes):
  """Creates the final MLP network combining each of the ViTs.
  
  Creates a Keras model with two layers combining the predicted label from each of the 
  Vision Transformers in the model. Uses the softmax or sigmoid activation function for final
  layer depending on the number of classes. This is different than the ViTs which do not output
  a probability distribution but instead output logits.

  Args:
    num_classse: The number of classes used.
  """
  activation = "softmax"
  if num_classes == 2:
    num_classes -= 1
    activation = "sigmoid"
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation=activation)
  ])
  # may want to compile
  return model

def get_ensemble(input_shape, feature_extractor, vits):
  """Creates an ensemble model given a feature extractor and list of ViTs.
  
  Creates a keras model and compiles it with the same metrics as the vits.
  Uses softmax activation function for final layer unlike ViTs which use logits.
  Args:
    input_shape: The shape of the image being ran. Does not include batch size.
    feature_extractor: Keras model to be used as a feature extractor.
    vits: List of Keras ViT models.
  """
  inputs = tf.keras.Input(shape=input_shape)
  x = feature_extractor(inputs)
  vals = []
  for vit in vits:
    for layer in vit.layers:
      layer.trainable = False
    vals.append(vit(x).logits)
  x = tf.keras.layers.Concatenate()(vals)
  x = tf.keras.layers.Dense(32, activation="relu")(x)
  outputs = tf.keras.layers.Dense(32, activation="softmax")(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  
  model.compile(
    optimizer="",
    loss="binary_crossentropy",
    metrics=["binary_accuracy", "AUC"]
  )
  return model

class SaveCallback(Callback):
  def __init__(self, save_path, save_metric, mode="max"):
    super().__init__()
    self.save_path = save_path
    self.metric = save_metric
    self.value = None
    if mode == "max":
      self.should_update = lambda new_value: True if self.value == None else new_value > self.value
    elif mode == "min":
      self.should_update = lambda new_value: True if self.value == None else new_value < self.value
    else:
      raise ValueError(f"mode should be either 'max' or 'min', not {mode}")
  
  def on_epoch_end(self, epoch, logs):
    new_value = logs[f"val_{self.metric}"]
    if self.should_update(new_value):
      print(f"Saving Model to {self.save_path}")
      self.model.save_pretrained(self.save_path)
      self.value = new_value
    return super().on_epoch_end(epoch, logs)

class EnsembleModel(tf.keras.Model):
  # should probably put arguments into a dict or something
  """Keras model for ensamble model

  Attributes:
    base_model: CNN feature extractor
    vit_models: List of trained ViT keras models
    concat: Concatenation layer combining ViT outputs
    meta1: hidden layer for meta network
    meta2: output of meta network
  """
  def __init__(self, feature_extractor, vits, meta):
    """Initializes ensemble model

    Args:
        num_classes: Number of classes in the data
        num_vit: Total number of ViTs being used
        base_model: CNN feature extractor keras model
        vit_models: List of trained ViT keras models
    """
    super().__init__()
    self.feature_extractor = feature_extractor
    for layer in feature_extractor:
      layer.trainable = False
    self.vits = vits
    for vit in vits: # models are already trained
      for layer in vit.layers:
        layer.trainable = False
      
    self.concat = tf.keras.layers.Concatenate()
    self.meta = meta
    self.print = True

  def call(self, inputs, training=False):
    """keras model method used to call the model

    Args:
        inputs: Input tensors to the model
        training: Boolean of if the model is being trained. Defaults to False.
    """
    x = self.feature_extractor(inputs)
    vals = []
    for model in self.vits:
      val = model(x).logits
      if self.print:
        tf.print(val)
      vals.append(val)
    self.print = False
    x = self.concat(vals)
    x = self.meta(x)
    return x