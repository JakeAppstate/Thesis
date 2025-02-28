import tensorflow as tf
#could maybe move imports to load method?
from  tensorflow.keras.applications import InceptionV3, ResNet50V2, ConvNeXtBase
from transformers import AutoConfig, TFAutoModel

# TODO
# 1. Add more ViT types including SWIN
# 2. Create model checkpoints for training

class VisionTransformer():
  def __init__(self, name, hf_id, cnn_data):
    self.name = name
    img_size, num_channels, num_classes = cnn_data
    config = AutoConfig.from_pretrained(hf_id, image_size = img_size, patch_size = 1, num_channels = num_channels, num_labels=num_classes)
    model = TFAutoModel.from_config(config)
    model.compile(
      optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5),
      loss = tf.keras.loss.BinaryCrossentropy(),
      metrics = ["binary_accuracy", "AUC"],
    )
    self.model = model

  def train(self, ds, epochs, save_dir="models"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=f"{save_dir}/{self.name}.keras",
      monitor='AUC',
      mode='max',
      save_best_only=True,
      save_weights_only=False
    )
    ds = ds.map(lambda img, label: (self.base_model(img), label)) #TODO add threads
    self.model.fit(ds, epochs=epochs, callbacks=[model_checkpoint_callback])

  @staticmethod
  def get_keras_from_name(name, save_dir):
    return tf.keras.models.load_model(f"{save_dir}/{name}.keras")

class CNN_ViT_Builder():
  """Creates the necessary components and instantiates the CNN_ViT_Ensemble class

  Creates specified CNN and ViTs
  Trains ViTs before ensembling
  Adds layer to connect CNN and ViT
  Creates an instanse of the CNN_ViT_Ensemble class
  
  Attributes:
    num_classes: Number of different classes of data.
    base_model: CNN combinded with reshaping layer.
    vit_models: List of ViT models from Hugging Face.
    num_vits: Number of ViTs used.
  """

  def __init__(self, input_shape, num_classes, cnn_type):
    """Initializes the object and creates keras models

    Args:
        input_shape: Tuple defining the input shape of the model
        num_classes: Int defining the number of classes
        cnn_type: String specifying the CNN to be used e.g. InceptionV3
        vit_types: List of strings specifying the ViTs to be used e.g. SWIN
            duplicate entries result in 2 identical models being trained
    """
    self.num_classes = num_classes
    cnn_model = self._get_cnn_model(cnn_type, input_shape)
    _, w, h, c = cnn_model.output_shape
    permute = tf.keras.layers.Permute((3, 1, 2)) # move channels to the front
    self.base_model = tf.keras.Sequential([cnn_model, permute])

    self.cnn_img_size = w
    self.cnn_channels = c

    self.vits = []

  def get_cnn_data(self):
    return (self.cnn_img_size, self.cnn_channels, self.num_classes)
  
  def add_vit(self, name):
    self.vits.append(VisionTransformer.get_keras_from_name(name))

  def train(self, ds, num_epochs, save_dir):
    """Trains the ViTs before ensembling

    Args:
        ds: Tf.data.Dataset object containing the training data
        num_epochs: Int specifying the number of epochs trained
    """
    # image, label = ds
    ds = ds.map(lambda img, label: (self.base_model(img), label)) #TODO add threads
    for model in self.vit_models:
        # TODO add checkpoints
        path = f"{save_dir}/"
        tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True)
        model.fit(ds, epochs=num_epochs)

  def build(self):
    """Creates the CNN_ViT_Ensamble model"""
    model = CNN_ViT_Ensemble(self.num_classes, self.num_vits, self.base_model, self.vit_models)
    return model

  def _get_cnn_model(self, model_name, input_shape):
    """Private method that converts CNN model name to keras model

    Args:
        model_name: String containing the name of the model to be created
        input_shape: The input shape of the CNN
    """
    if(model_name == "convnext"):
      return ConvNeXtBase(include_top = False, input_shape = input_shape)
    if(model_name == "inception_v3"):
      return InceptionV3(include_top = False, input_shape = input_shape)
    if(model_name == "resnet50"):
      return ResNet50V2(include_top = False, input_shape = input_shape)
    raise ValueError(f"model_name: {model_name} is not a valid input")

  def _get_vit_model(self, model_name, img_size, num_channels, num_classes):
    """Private method that loads the keras model for the given ViT

    Args:
      model_name: Name of the ViT e.g. DeiT
      img_size: size of input image to the ViT
      num:channels: number of channels of the input image
      num_classes: number of classes of the data
    """
    if(model_name == "ViT"):
      # config = ViTConfig(image_size = img_size, patch_size = 1, num_channels = num_channels, num_labels=num_classes)
      name = "google/vit-base-patch16-224"
      # return TFViTForImageClassification.from_pretrained(name, config = config, ignore_mismatched_sizes=True)
    elif(model_name == "DeiT"):
      # config = DeiTConfig(image_size = img_size, patch_size = 1, num_channels = num_channels, num_labels=num_classes)
      name = "facebook/deit-base-distilled-patch16-224"
      # return TFDeiTForImageClassification.from_pretrained(name, config = config, ignore_mismatched_sizes=True)
    # add SWIN
    elif model_name == "SWIN":
        name = "microsoft/swin-tiny-patch4-window7-224"
    else:
      raise ValueError(f"Vision Transformer {model_name} is not a valid entry")
    config = AutoConfig.from_pretrained(name, image_size = img_size, patch_size = 1, num_channels = num_channels, num_labels=num_classes)
    model = TFAutoModel.from_config(config)
    return model

class CNN_ViT_Ensemble(tf.keras.Model):
  # should probably put arguments into a dict or something
  """Keras model for ensamble model

  Attributes:
    base_model: CNN feature extractor
    vit_models: List of trained ViT keras models
    concat: Concatenation layer combining ViT outputs
    meta1: hidden layer for meta network
    meta2: output of meta network
  """
  def __init__(self, num_classes, num_vits, base_model, vit_models, **kwargs):
    """Initializes ensemble model

    Args:
        num_classes: Number of classes in the data
        num_vit: Total number of ViTs being used
        base_model: CNN feature extractor keras model
        vit_models: List of trained ViT keras models
    """
    super().__init__(**kwargs)
    # h, w, c = scale_shape
    # self.resize = tf.keras.layers.Resizing(h, w)
    # self.rescale = tf.keras.layers.Rescaling(scale = 1/255)
    # self.base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=scale_shape)
    self.base_model = base_model
    for layer in self.base_model.layers: # Do not want to change weights of feature extractor
      layer.trainable = False

    for model in vit_models: # models are already trained
      for layer in model.layers:
        layer.trainable = False

    self.vit_models = vit_models
    self.concat = tf.keras.layers.Concatenate()
    int_size = num_vits * num_classes // 2
    self.meta1 = tf.keras.layers.Dense(int_size, activation="relu")
    self.meta2 = tf.keras.layers.Dense(num_classes, activation="softmax")

  def call(self, inputs, training=False):
    """keras model method used to call the model

    Args:
        inputs: Input tensors to the model
        training: Boolean of if the model is being trained. Defaults to False.
    """
    x = inputs
    x = self.base_model(x)
    vals = []
    for model in self.vit_models:
      vals.append(model(x).logits)
    x = self.concat(vals)
    x = self.meta1(x)
    x = self.meta2(x)
    return x

