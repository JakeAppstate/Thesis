import csv
import tensorflow as tf
from tensorflow.data import Dataset
import tf_clahe

@tf.function
def load_jpg(filename, img_size):
    """Loads an image tensor from a given file"""
    raw = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, img_size)
    return img

def get_folder(id):
    #TODO implement when all data is downloaded
    pass

def get_filepath(name, data_dir):
    """Converts an image name to file path"""
    # TODO Add functionality for multiple folders
    # could move all files into one folder? use glob?
    return f"{data_dir}/0/{name}"

def get_csv_ds(filepath, data_dir, data_col, label_col):
    """Gets a Dataset consising of the filenames listed in the csv_file and the binary label."""
    with open(filepath, 'r') as file:
        csv_iter = csv.reader(file, delimiter=';')

        path_iter = map(lambda row: (get_filepath(row[0], data_dir), row[1]), csv_iter)
        valid_iter = filter(lambda row: tf.io.gfile.exists(row[0]), path_iter)
        path, label = zip(*valid_iter) #unzip
        return Dataset.from_tensor_slices((list(path), list(label)))

def get_dataset(csv_file, data_dir, img_size):
    """Gets a Dataset consiting of the loaded image and binary label"""
    csv_ds = get_csv_ds(csv_file, data_dir)
    print_ds(csv_ds.take(1))
    img_ds = csv_ds.interleave(
            lambda x, y: Dataset.from_tensor_slices(([load_jpg(x, img_size)], [y])),
            num_parallel_calls=tf.data.AUTOTUNE
            )
    return img_ds

def get_preprocess_model(target_shape, yolo_path):
    """Rescales, normalizes, and augments images"""
    input = tf.keras.Input()
    x = tf.keras.layers.Resizing(w = 2_000, h=2_000)(input)
    x = tf.keras.layers.Lambda(tf_clahe.clahe)(x)
    x = tf.keras.layers.Rescaling(1./255)
    yolo = tf.keras.models.load_model(yolo_path)
    yolo_x = tf.keras.Resizing(w = 640, h=640)(x)
    b_x, b_y, b_w, b_h = yolo(yolo_x) * 2_000 / 640 # scale to non yolo image
    x = tf.keras.layers.Lambda(lambda x: tf.image.crop_to_bounding_box(x, b_y, b_x, b_h, b_w))
    x = tf.keras.layers.Resizing(w=target_shape[0], h=target_shape[1])
    return tf.keras.Model(inputs=input, outputs=x)

def get_augmentaion():
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
def oversample(ds, ratio):
    ds_neg = ds.filter(lambda x, y: y == 0)
    num_neg = ds_neg.cardinality()
    ds_pos = ds.filter(lambda x, y: y == 1)
    num_pos = ds_pos.cardinality()
    n = num_neg / ratio - num_neg #total number of positive samples needed for given ratio
    
    # use cache to use the base images for each augmentaion
    ds_pos = ds_pos \
        .cache() \
        .map(get_augmentaion) \
        .repeat() \
        .take(n)
    return ds_neg.concatenate(ds_pos).shuffle()

def load_yolo(path):
    return tf.keras.models.load_model(path)

@tf.function
def apply_clahe(img, label):
    img = tf_clahe.clahe(img)
    return img, label

def print_ds(ds):
    for item in ds:
        print(item)

if __name__ == "__main__":
    data_dir = "../data"
    ds = get_dataset(f"{data_dir}/labels.csv", data_dir, (1024, 1024))
    ds = ds.map(apply_clahe)

    print_ds(ds.take(1))
