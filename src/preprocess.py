import csv
import tensorflow as tf
from tensorflow.data import Dataset

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

def get_filepath(id, data_dir):
    """Converts an image id to file path"""
    # TODO Add functionality for multiple folders
    # could move all files into one folder? use glob?
    return f"{data_dir}/0/{id}.JPG"

def get_csv_ds(filepath, data_dir):
    """Gets a Dataset consising of the filenames listed in the csv_file and the binary label."""
    with open(filepath, 'r') as file:
        csv_iter = csv.reader(file, delimiter=';')
        path_iter = map(lambda row: (get_filepath(row[0], data_dir), row[1]), csv_iter)
        valid_iter = filter(lambda row: tf.io.gfile.exists(row[0]), path_iter)
        path, label = zip(*valid_iter)
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

def preprocess_model(target_shape):
    """Rescales, normalizes, and augments images"""
    w, h, c = target_shape
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1./255),
        # tf.keras.layers.Resizing(width=1000, height=1000),
        tf.keras.layers.CenterCrop(width=2*w, height=2*h),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(.05, .05),
        tf.keras.layers.RandomZoom(.05),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Resizing(width=w, height=h)
    ])
    return model

def print_ds(ds):
    for item in ds:
        print(item)

if __name__ == "__main__":
    data_dir = "../data"
    ds = get_dataset(f"{data_dir}/labels.csv", data_dir, (1024, 1024))
    print_ds(ds.take(1))
