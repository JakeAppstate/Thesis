import csv
import tensorflow as tf # type: ignore
from tensorflow.data import Dataset # type: ignore
# import tf_clahe
import cv2 # type: ignore

@tf.function
def load_jpg(filename, img_size):
    """Loads an image tensor from a given file"""
    raw = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, img_size)
    return tf.cast(img, tf.uint8)

def get_folder(id):
    #TODO implement when all data is downloaded
    pass

def get_filepath(name, data_dir):
    """Converts an image name to file path"""
    # TODO Add functionality for multiple folders
    # could move all files into one folder? use glob?
    return f"{data_dir}/0/{name}.JPG"

def get_csv_ds(filepath, data_dir):
    """Converts csv file into tf dataset.

    Loads dataset in the form of (filepath, label) for each image id found in directory.
    Args:
        filepath: Full path to the csv file to be loaded.
        data_dir: Root folder of data location.
    """
    with open(filepath, 'r') as file:
        csv_iter = csv.reader(file, delimiter=';')

        path_iter = map(lambda row: (get_filepath(row[0], data_dir), int(row[1] == "RG")), csv_iter)
        valid_iter = filter(lambda row: tf.io.gfile.exists(row[0]), path_iter)
        path, label = zip(*valid_iter) #unzip
        path, label = list(path), list(label)
        return Dataset.from_tensor_slices((path, label))

def get_datasets(data_dir, csv_file, img_size, target_shape, 
                 train_split, val_split, sample_ratio, yolo_path):
    """Gets final train, validation and test datasets.
    
    Loads the dataset from the csv file and loads the images. Preprocess the images using
    the preproces function and counts the number of negative and positive labels. Oversamples to 
    get dataset to desired class ratio and splits into train, validation and test sets.
    
    Args:
        data_dir: Main directory the data is stored including csv file and images.
        csv_file: The name of the csv_file. Should be located in the root of data_dir.
        img_size: Initial size for the images to be loaded as.
        target_shape: Desired shape for images after preprocessing is applied.
        train_split: Percent of dataset to be reserved for training and validation.
        validation_split: Persent of the training data to be used for validation.
        sample_ratio: Desired ratio for oversampling.
        yolo_path: Filepath to stored YOLOv8 model.
    """
    # load data from csv file
    filepath = f"{data_dir}/{csv_file}"
    csv_ds = get_csv_ds(filepath, data_dir)
    tf.print("Csv size:", csv_ds.cardinality())
    csv_ds = csv_ds.shuffle(1_000, reshuffle_each_iteration=False)

    # count number of each class before loading images to increase performance
    train_ds, test_ds = split_ds(csv_ds, train_split)
    tf.print("Train before mapping size:", train_ds.cardinality())
    tf.print("Test cardinality:", test_ds.cardinality())
    reduce_fun = lambda state, data: tf.cond(data[1] == 1, 
                                             lambda: (state[0], state[1] + 1), 
                                             lambda: (state[0] + 1, state[1]))
    counts = train_ds.reduce((0, 0), reduce_fun)
    
    # load and preprocess images
    f = lambda x, y: (preprocess(load_jpg(x, img_size), img_size, target_shape, yolo_path), y)
    train_ds = train_ds.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    tf.print("Train after mapping size:", train_ds.cardinality())

    # oversamples train and validation data
    train_ds = oversample(train_ds, sample_ratio, counts)
    tf.print("Train after oversampling size:", train_ds.cardinality())
    train_ds, val_ds = split_ds(train_ds, 1 - val_split)
    tf.print("Final train size:", train_ds.cardinality())
    tf.print("Final validation size:", val_ds.cardinality())

    return train_ds, val_ds, test_ds

def preprocess(img, img_size, target_shape, yolo_path):
    """"Preprocessing to be applied to an image.
    
    TF operations to preprocess a loaded image. Applies CLAHE contrast enhancement, 
    normalizes image, and crops at the ROI.
    Args:
        img: Image to be preprocessed.
        img_size: The width and height of img.
        target_shape: The desired shape of the preprocessed image. In the form (w, h, c).
        yolo_path: Filepath to stored YOLOv8 model.
    """
    img = apply_clahe(img)
    img /= 255.
    yolo_img = tf.image.resize(img, (640, 640))
    b_x, b_y, b_w, b_h = yolo(yolo_img, yolo_path, *img_size)
    w, h, _ = target_shape
    # convert center of box to top-left
    crop_x = tf.clip_by_value(b_x - w // 2, 0, img_size[0] - w)
    crop_y = tf.clip_by_value(b_y - h // 2, 0, img_size[1] - h)
    img = tf.image.crop_to_bounding_box(img, crop_y, crop_x, h, w)
    return img

def get_augmentaion():
    """Gets a Keras model used for data augmentation."""
    #https://link.springer.com/article/10.1007/s10462-023-10453-z#Sec23
    #TODO Use methods from paper
    model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(.05, .05),
        tf.keras.layers.RandomZoom(.05),
        tf.keras.layers.RandomContrast(0.1),
    ])
    return model

#TODO Look into balanced bagging
def oversample(ds, ratio, counts):
    """Oversamples the dataset to have output have negative to positve
    classes being equal to ratio.
    
    Uses data augemntation to oversample the positive class until the desired ratio is met.
    Cardinality has to be asserted since filter method is used. This also requires prior knowledge
    of the origional counts of the classes.
    
    Args:
        ds: The dataset to be oversampled.
        ratio: The desired ratio of negative to positive.
        counts: The number of negative and positive cases in ds. 
        Given in the form of a tuple (neg, pos).
    """
    num_neg, num_pos = counts
    num_neg = tf.cast(num_neg, tf.float32)
    num_pos = tf.cast(num_pos, tf.float32)
    tf.print(num_neg, num_pos)
    
    ds_neg = ds.filter(lambda x, y: y == 0)
    ds_pos = ds.filter(lambda x, y: y == 1)

    #total number of positive samples needed for given ratio
    n = tf.cast(num_neg / ratio - num_neg, tf.int64) 
    # use cache to use the base images for each augmentaion
    aug = get_augmentaion()
    ds_pos = ds_pos \
        .cache() \
        .map(lambda x, y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .repeat() \
        .take(n)
    
    ds = ds_neg.concatenate(ds_pos).shuffle(1_000)
    ds = ds.apply(tf.data.experimental.assert_cardinality(n + tf.cast(num_neg, tf.int64)))
    tf.print("Size after oversampling:", ds.cardinality())
    return ds

def split_ds(ds, ratio):
    """Splits the tf dataset into two parts.
    
    Dataset is split into two parts with the first being ratio percent of the
    original dataset and the second being 1 - ratio.
    Args:
        ds: The dataset to be split.
        ratio: Percent of the original dataset the first output should be.
    """
    n = tf.cast(ds.cardinality(), tf.float32)
    n_train = tf.cast(n * ratio, tf.int64)
    train_ds = ds.take(n_train)
    test_ds = ds.skip(n_train)
    return train_ds, test_ds

def yolo(img, path, orig_w, orig_h):
    """Runs YOLO model on image.
    
    Loads YOLO model from the disk and gets the bounding box with the highest probability.
    Returns the center of the box mapped to the original image.

    Args:
        img: Image to be run. Expected to be of size (640, 640, 3).
        path: Filepath to YOLO model.
        orig_w: Width of original image
        orig_h: Height of original image.
    """
    yolo = tf.saved_model.load(path)
    img = tf.expand_dims(img, 0)
    output = yolo(img) # expects [Batch, 640, 640 , 3]
    output = tf.squeeze(output)
    i = tf.argmax(output[4, :])
    x, y, w, h = tf.unstack(tf.cast(output[:4, i], tf.int32), num=4)
    #scale to original image
    x *= orig_w // 640
    w *= orig_w // 640
    y *= orig_h // 640
    h *= orig_h // 640
    return x, y, w, h

def load_yolo(path):
    """Loads YOLO model from path.
    
    Args:
        path: File location of yolo model.
    """
    return tf.keras.models.load_model(path)

# can implement later if needed
@tf.py_function(Tout = tf.float32)
def clahe(img):
    """Implementation of CLAHE.
    
    Uses open-cv library to enhance image. Changes color-space to lab and
    appies CLAHE on Lightness channel, then converts image back to RGB.
    
    Args:
        img: Image to be enhanced. using CLAHE.
    """
    img = img.numpy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    new_img = cv2.merge((l, a, b))
    new_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return tf.convert_to_tensor(new_img, dtype=tf.float32)

def apply_clahe(img):
    """Applies CLAHE contrast enhancement to an image.
    
    Wrapper function to run CLAHE for the tensorflow graph. 
    Asserts that the shape stays constant through the transformation.
    
    Args:
        img: Image to be enhanced. using CLAHE.
    """
    img_shape = img.shape
    img = clahe(img)
    img.set_shape(img_shape)
    return img
