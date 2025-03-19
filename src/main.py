import tensorflow as tf

from preprocess import get_datasets
from model import get_feature_extractor, get_vit, EnsembleModel, SaveCallback

#Preprocessing
DATA_DIR = "./data"
CSV_FILE = "labels.csv"
IMG_SIZE = (2_000, 2_000)
OVERSAMPLE_RATIO = 0.75
YOLO_PATH = "./models/yolo"
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.2

# Model Info
INPUT_SHAPE = (512, 512, 3)
NUM_ClASSES = 2
SAVE_DIR = "./models"
CNN = "Inceptionv3"
# Using new models
VITS = [
    ("ViT", "google/vit-base-patch16-224"),
    ("DeiT","facebook/deit-base-distilled-patch16-224"),
    ("MobileViT", "apple/mobilevit-small")
    # ("SWIN", "microsoft/swin-tiny-patch4-window7-224")
]

# Using saved models
# VITS = [
#     ("ViT", f"{SAVE_DIR}/vits/ViT"),
#     ("DeiT",  f"{SAVE_DIR}/vits/DeiT")
# ]
HF_KWARGS = {
    "patch_size": 1,
    "problem_type": "single_label_classification",
    "num_labels": 1
}

#Training
BATCH_SIZE = 32
VIT_EPOCHS = 10
FINAL_EPOCHS = 10
HAVE_TRAINED = False

def main():
    # DEBUG
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    train_ds, val_ds, test_ds = get_datasets(DATA_DIR, CSV_FILE, IMG_SIZE, INPUT_SHAPE,
                                             TRAIN_SPLIT, VAL_SPLIT, OVERSAMPLE_RATIO, YOLO_PATH)
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # strategy = tf.distribute.MirroredStrategy()

    # Get CNN and extract features
    feature_extractor = get_feature_extractor(CNN, INPUT_SHAPE)
    _, c, w, h = feature_extractor.output_shape
    vit_train_ds = train_ds.map(lambda x, y: (feature_extractor(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
    vit_val_ds = val_ds.map(lambda x, y: (feature_extractor(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
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
        vits.append(vit)
    
    #Get, compile, and train final model
    print("Running final model")
    ensemble = EnsembleModel(NUM_ClASSES, feature_extractor, vits)
    ensemble.compile(
        optimizer="adamw",
        loss="binary_crossentropy",
        metrics=["binary_accuracy", "AUC"]
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"{SAVE_DIR}/final.keras",
        monitor='val_auc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
    )
    ensemble.fit(train_ds,
                 epochs = FINAL_EPOCHS,
                 validation_data = val_ds,
                 callbacks = [ckpt],
                 verbose = 2
                 )
    
    ensemble.evaluate(
        test_ds
    )
    

if __name__ == "__main__":
    main()
