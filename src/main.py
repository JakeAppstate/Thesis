import tensorflow as tf # type: ignore

from preprocess import get_datasets
from model import get_feature_extractor, get_vit, get_meta, call_all_vits, EnsembleModel, SaveCallback

# Seed
SEED = 560

#Preprocessing
DATA_DIR = "./data"
CSV_FILE = "labels.csv"
IMG_SIZE = (2_000, 2_000)
OVERSAMPLE_RATIO = 0.75
YOLO_PATH = "./models/yolo"
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.2

# Model Info
INPUT_SHAPE = (560, 560, 3)
NUM_ClASSES = 2
SAVE_DIR = "./models"
CNN = "Inceptionv3"
# Using new models
# VITS = [
#     ("ViT", "google/vit-base-patch16-224"),
#     ("DeiT","facebook/deit-base-distilled-patch16-224"),
#     ("SWIN", "microsoft/swin-tiny-patch4-window7-224")
# ]

# Using saved models
VITS = [
    ("ViT", f"{SAVE_DIR}/vits/ViT"),
    ("DeiT",  f"{SAVE_DIR}/vits/DeiT"),
    ("SWIN", f"{SAVE_DIR}/vits/SWIN")
]

HF_KWARGS = {
    "patch_size": 1,
    "problem_type": "single_label_classification",
    "num_labels": 1,
    "window_size": 2 # For SWIN if needed
}

#Training
BATCH_SIZE = 32
VIT_EPOCHS = 10
FINAL_EPOCHS = 10
HAVE_TRAINED = True

def main():
    # DEBUG
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()
    tf.random.set_seed(SEED)

    train_ds, val_ds, test_ds = get_datasets(DATA_DIR, CSV_FILE, IMG_SIZE, INPUT_SHAPE,
                                             TRAIN_SPLIT, VAL_SPLIT, OVERSAMPLE_RATIO, YOLO_PATH)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # strategy = tf.distribute.MirroredStrategy()

    # Get CNN and extract features
    feature_extractor = get_feature_extractor(CNN, INPUT_SHAPE)
    _, c, w, h = feature_extractor.output_shape
    map_fun = lambda x, y: (feature_extractor(x), y)
    vit_train_ds = train_ds.map(map_fun, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
    vit_val_ds = val_ds.map(map_fun, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Get and train ViTS
    vits = []
    for name, url in VITS:
        print("Running", name)
        vit = get_vit(url, w, c, HF_KWARGS)
        if not HAVE_TRAINED:
            clbk = SaveCallback(f"{SAVE_DIR}/vits/{name}", "auc")
            vit.fit(vit_train_ds,
                    epochs=VIT_EPOCHS,
                    validation_data = vit_val_ds,
                    callbacks=[clbk],
                    verbose = 2
                    )
        # maybe add test set evaluation for each model?
        vits.append(vit)
    
    #Get, compile, and train final model
    print("Training meta network")
    map_fun = lambda x, y: (call_all_vits(x, vits), y)
    meta_train_ds = vit_train_ds.map(map_fun, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
    meta_val_ds = vit_val_ds.map(map_fun, num_parallel_calls=tf.data.AUTOTUNE)

    meta = get_meta(NUM_ClASSES)
    compile_dict = {
        "optimizer": "adamw",
        "loss": "binary_crossentropy",
        "metrics": ["binary_accuracy", "AUC"]
    }
    meta.compile_from_config(compile_dict)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"{SAVE_DIR}/meta.keras",
        monitor='val_auc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
    )
    meta.fit(meta_train_ds,
             epochs = FINAL_EPOCHS,
             validation_data = meta_val_ds,
             callbacks = [ckpt],
             verbose = 2
            )
    print("Running final model")
    final_model = EnsembleModel(feature_extractor, vits, meta)
    final_model.compile_from_config(compile_dict)
    final_model.evaluate(test_ds)
    

if __name__ == "__main__":
    main()
