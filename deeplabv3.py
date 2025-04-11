"""-------------------------------------------------IMPORT LIBRARIES-------------------------------------------------"""
import os

import keras
import requests
from zipfile import ZipFile
import glob
from dataclasses import dataclass, field

import random
import numpy as np
import cv2

import tensorflow as tf
import keras_cv

import matplotlib.pyplot as plt
from keras.src.utils import load_img, img_to_array
import warnings

warnings.filterwarnings("ignore")

# We followed this tutorial :
# https://github.com/spmallick/learnopencv/blob/master/Semantic-Segmentation-using-KerasCV-with-DeepLabv3-Plus/Segmenation_deeplabv3_plus_resnet50v2.ipynb
"""--------------------------------------------------- CONFIG---------------------------------------------------"""
def system_config(SEED_VALUE):
    # Set python `random` seed.
    # Set `numpy` seed
    # Set `tensorflow` seed.
    random.seed(SEED_VALUE)
    tf.keras.utils.set_random_seed(SEED_VALUE)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_USE_CUDNN'] = "true"
    tf.keras.backend.clear_session()

system_config(SEED_VALUE=42)


class DatasetConfig:
    IMAGE_SIZE:        tuple = (128, 128)
    BATCH_SIZE:          int = 8
    NUM_CLASSES:         int = 7
    # BRIGHTNESS_FACTOR: float = 0.3
    # CONTRAST_FACTOR:   float = 0.3


class TrainingConfig:
    MODEL:           str = "resnet50_v2_imagenet"
    EPOCHS:          int = 5
    LEARNING_RATE: float = 0.02
    CKPT_DIR: str = os.path.join("checkpoints_" + "_".join(MODEL.split("_")[:2]),
                                 "deeplabv3_plus_" + "_".join(MODEL.split("_")[:2]) + ".weights.h5")
    LOGS_DIR: str = "logs_" + "_".join(MODEL.split("_")[:2])




"""---------------------------------------------------LOAD DATASET---------------------------------------------------"""
train_config = TrainingConfig()
dataset_config = DatasetConfig()

data_im = glob.glob(os.path.join("dataset", "*.jpg"))
data_masks = glob.glob(os.path.join("dataset", "*.png"))
# Shuffle the data paths before data preparation.
zipped_data = list(zip(data_im, data_masks))
random.shuffle(zipped_data)
data_im, data_masks = zip(*zipped_data)
data_im = list(data_im)
data_masks = list(data_masks)
org_data = tf.data.Dataset.from_tensor_slices((data_im, data_masks))

SPLIT_RATIO = 0.3
# Determine the number of validation samples
NUM_VAL = int(len(data_im) * SPLIT_RATIO)

train_data = org_data.skip(NUM_VAL)
val_data = org_data.take(NUM_VAL)

# print(train_data.cardinality().numpy())

data_im_test = glob.glob(os.path.join("test", "*.jpg"))
data_masks_test = glob.glob(os.path.join("test", "*.png"))
test_data = tf.data.Dataset.from_tensor_slices((data_im_test, data_masks_test))

def read_image_mask(image_path, mask=False, size = DatasetConfig.IMAGE_SIZE):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.io.decode_image(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=size, method = "bicubic")

        image_mask = tf.zeros_like(image)
        cond = image >=200
        updates = tf.ones_like(image[cond])
        image_mask = tf.tensor_scatter_nd_update(image_mask, tf.where(cond), updates)
        image = tf.cast(image_mask, tf.uint8)

    else:
        image = tf.io.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=size, method = "bicubic")
        image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.float32)

    return image

def load_data(image_list, mask_list, mask=True):
    image = read_image_mask(image_list)
    mask  = read_image_mask(mask_list, mask=mask)
    return {"images":image, "segmentation_masks":mask}

train_ds = train_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_data.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)



"""---------------------------------------------------MASK CONFIG---------------------------------------------------"""
id2color = {
    0: (0, 255, 255),  # urban
    1: (255, 255, 0),  # agri
    2: (255, 0, 255),  # rangeland
    3: (0, 255, 0), # forest
    4: (0, 0, 255), # water
    5: (255, 255, 255), # barren
    6: (0, 0, 0),  # Background
}

def color_to_index(mask):
    # Create an empty index mask with the same spatial dimensions
    index_mask = tf.zeros(tf.shape(mask)[:-1], dtype=tf.int32)

    # For each class, create a boolean mask and assign the class index
    for class_idx, color in id2color.items():
        # Create a boolean mask for pixels matching this color
        color_tensor = tf.constant(color, dtype=mask.dtype)
        matches = tf.reduce_all(tf.equal(mask, color_tensor), axis=-1)

        # Update the index mask where matches are found
        index_mask = tf.where(matches, class_idx, index_mask)

    return index_mask
def unpackage_inputs(inputs, one_hot=False):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    # If masks are in RGB format, convert to indices first
    if segmentation_masks.shape[-1] == 3:  # RGB format
        segmentation_masks = color_to_index(segmentation_masks)

    if one_hot:
        num_classes = len(id2color)
        indices = tf.cast(segmentation_masks, tf.int32)
        segmentation_masks = tf.one_hot(indices, depth=num_classes)
        segmentation_masks = tf.squeeze(segmentation_masks, axis=-2)
        # print(segmentation_masks.shape)
    return images, segmentation_masks




"""------------------------------------------------HISTOGRAM MATCHING------------------------------------------------"""
# https://medium.com/data-science/histogram-matching-ee3a67b4cbc1
def print_img(img, histo_new, histo_old, index, L):
    dpi = 80
    width = img.shape[0]
    height = img.shape[1]
    figsize = (width) / float(dpi), (height*4) / float(dpi)
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1,1]}, figsize=figsize)

    fig.suptitle("Enhanced Image with L:" + str(L))
    axs[0].title.set_text("Enhanced Image")
    axs[0].imshow(img, vmin=np.amin(img), vmax=np.amax(img), cmap='gray')

    axs[1].title.set_text("Equalized histogram")
    axs[1].plot(histo_new, color='#f77f00')
    axs[1].bar(np.arange(len(histo_new)), histo_new, color='#003049')

    axs[2].title.set_text("Main histogram")
    axs[2].plot(histo_old, color='#ef476f')
    axs[2].bar(np.arange(len(histo_old)), histo_old, color='#b7b7a4')
    plt.tight_layout()


def print_histogram(_histrogram, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(_histrogram, color='#ef476f')
    plt.bar(_histrogram, _histrogram, color='#b7b7a4')
    plt.ylabel('Number of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("hist_" + name)


def generate_histogram(img, do_print, index):
    if len(img.shape) == 3:  # img is colorful
        gr_img = np.mean(img, axis=-1)
    else:
        gr_img = img
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([256])

    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value] += 1
    '''normalize Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1])
    if do_print:
        print_histogram(gr_hist, name="neq_"+str(index), title="Normalized Histogram")
    return gr_hist, gr_img


def equalize_histogram(img, histo, L, index=0):
    eq_histo = np.zeros_like(histo)
    en_img = np.zeros_like(img)
    for i in range(len(histo)):
        eq_histo = int((L - 1) * np.sum(histo[0:i]))
    print_histogram(eq_histo, name="eq_"+str(index), title="Equalized Histogram")
    '''creating new histogram'''
    hist_img, _ = generate_histogram(en_img, do_print=True, index=index)
    print_img(img=en_img, histo_new=hist_img, histo_old=histo, index=str(index), L=L)
    return eq_histo

def find_value_target(val, target_arr):
    key = np.where(target_arr == val)[0]

    if len(key) == 0:
        key = find_value_target(val+1, target_arr)
    return key



"""---------------------------------------------------VISUALIZATION--------------------------------------------------"""
def num_to_rgb(num_arr, color_map=id2color):
    if len(num_arr.shape) == 3 and num_arr.shape[2] > 1:
        class_indices = np.argmax(num_arr, axis=2)
    else:
        class_indices = np.squeeze(num_arr)
    output = np.zeros(class_indices.shape + (3,))
    for k in color_map.keys():
        output[class_indices == k] = color_map[k]

    return output.astype(np.uint8)

# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
    alpha = 1.0 # Transparency for the original image.
    beta  = 0.5 # Transparency for the segmentation map.
    gamma = 0.0 # Scalar added to each sum.

    image = image.astype(np.uint8)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def display_image_and_mask(data_list, title_list, figsize, color_mask=False):
    # Create RGB segmentation map from grayscale segmentation map.
    rgb_gt_mask = num_to_rgb(data_list[1])
    mask_to_overlay = rgb_gt_mask
    if len(data_list) == 3:
        rgb_pred_mask = num_to_rgb(data_list[-1])
        mask_to_overlay = rgb_pred_mask
    # Create the overlayed image.
    overlayed_image = image_overlay(data_list[0], mask_to_overlay)
    data_list.append(overlayed_image)
    fig, axes = plt.subplots(nrows=1, ncols=len(data_list), figsize=figsize)
    for idx, axis in enumerate(axes.flat):
        axis.set_title(title_list[idx])
        if title_list[idx] == "GT Mask":
            if color_mask:
                axis.imshow(rgb_gt_mask)
            else:
                axis.imshow(data_list[1], cmap="gray")
        elif title_list[idx] == "Pred Mask":
            if color_mask:
                axis.imshow(rgb_pred_mask)
            else:
                axis.imshow(data_list[-1], cmap="gray")
        else:
            axis.imshow(data_list[idx])
        axis.axis('off')


plot_train_ds = train_ds.map(unpackage_inputs).batch(3)
image_batch, mask_batch = next(iter(plot_train_ds.take(1)))

titles = ["GT Image", "GT Mask", "Overlayed Mask"]

n=0
for image, gt_mask in zip(image_batch, mask_batch):
    gt_mask = tf.squeeze(gt_mask, axis=-1).numpy()
    display_image_and_mask([image.numpy().astype(np.uint8), gt_mask],
                           title_list=titles,
                           figsize=(16,6),
                           color_mask=True)
    n += 1
    plt.savefig(f"image{n}.png")

# Histogram matching
L =50
gr_img_arr = []
gr_hist_arr = []
eq_hist_arr = []
index = 0
for img, _ in zip(image_batch, mask_batch):
    hist_img, gr_img = generate_histogram(img, do_print=True, index=index)
    gr_hist_arr.append(hist_img)
    gr_img_arr.append(gr_img)
    eq_hist_arr.append(equalize_histogram(gr_img, hist_img, L))
    index += 1
"""------------------------------------------------DATA AUGMENTATION-------------------------------------------------"""
augment_fn = tf.keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras.layers.RandomCrop(128, 128),
    ]
)

augment_val = tf.keras.Sequential(
    [
        keras.layers.CenterCrop(128, 128)
    ]
)

train_dataset = (
                train_ds.shuffle(DatasetConfig.BATCH_SIZE)
                .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(DatasetConfig.BATCH_SIZE)
                .map(lambda inputs: unpackage_inputs(inputs, one_hot=True))
                .prefetch(buffer_size=tf.data.AUTOTUNE)
)

valid_dataset = (
                val_ds.batch(DatasetConfig.BATCH_SIZE)
                .map(augment_val, num_parallel_calls=tf.data.AUTOTUNE)
                .map(lambda inputs: unpackage_inputs(inputs, one_hot=True))
                .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_dataset = (
                test_ds.batch(DatasetConfig.BATCH_SIZE)
                .map(lambda inputs: unpackage_inputs(inputs, one_hot=True))
                .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# USED ONLY WITH one_hot=False!!

# im_batch, aug_mask_batch = next(iter(train_dataset.take(1)))

# labels = ["GT Image", "GT Mask", "Overlayed Mask"]
#
# for idx, (image, gt_mask) in enumerate(zip(image_batch, aug_mask_batch)):
#     if idx > 1:
#         break
#     gt_mask = tf.squeeze(gt_mask, axis=-1).numpy()
#     display_image_and_mask([image.numpy().astype(np.uint8), gt_mask],
#                            title_list=titles,
#                            figsize=(16,6),
#                            color_mask=False)
#     n += 1
#     plt.savefig(f"image_aug{n}.png")



"""----------------------------------------------LOAD PRE-TRAINED MODEL----------------------------------------------"""
backbone = keras_cv.models.ResNet50V2Backbone.from_preset(preset = train_config.MODEL,
                                                          input_shape=DatasetConfig.IMAGE_SIZE+(3,),
                                                          load_weights = True)
for layer in backbone.layers:
    layer.trainable = False

model = keras_cv.models.segmentation.DeepLabV3Plus(
        num_classes=DatasetConfig.NUM_CLASSES, backbone=backbone,
    )
print(model.summary())

"""----------------------------------------------PARTIAL CROSS-ENTROPY-----------------------------------------------"""
# focal loss source: https://github.com/meng-tang/rloss/blob/1caa759e568db2c7209ab73e73ac039ea3d7101c/pytorch/pytorch-deeplab_v3_plus/utils/loss.py#L34
class partialCE(tf.keras.losses.Loss):
    def __init__(self, gamma=2, alpha=0.5, ignore_index=255, weight=None, batch_average=True):
        super(partialCE, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average

    def call(self, y_true, y_pred):
        # print("y_true shape", y_true.shape)
        # print("y_pred shape", y_pred.shape)
        n = tf.shape(y_pred)[0]

        mask = tf.not_equal(tf.argmax(y_true, axis=-1), self.ignore_index)
        # print("mask", mask.shape)
        mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        # print("exp", mask_expanded.shape)
        y_true_one_hot = y_true * mask_expanded
        # print("oneh", y_true_one_hot.shape)
        # Compute softmax cross entropy
        ce_loss = tf.keras.losses.categorical_crossentropy(
            y_true_one_hot,
            y_pred,
            from_logits=True
        )
        pt = tf.exp(-ce_loss)

        # focal weighting
        focal_weight = tf.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss

        # consider only valid pixels
        mask_float = tf.cast(mask, tf.float32)
        masked_focal_loss = focal_loss * mask_float
        loss = tf.reduce_sum(masked_focal_loss)

        # Normalize by the number of valid pixels
        num_valid_pixels = tf.reduce_sum(mask_float)
        loss = loss / (num_valid_pixels + tf.keras.backend.epsilon())
        if self.batch_average:
            loss = loss / tf.cast(n, tf.float32)

        return loss


"""-----------------------------------------------------CALLBACK-----------------------------------------------------"""
def get_callbacks(
    train_config,
    monitor="val_mean_iou",
    mode="max",
    save_weights_only=True,
    save_best_only=True,
):

    # Initialize tensorboard callback for logging.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=train_config.LOGS_DIR,
        histogram_freq=20,
        write_graph=False,
        update_freq="epoch",
    )


    # Update file path if saving best model weights.
    if save_weights_only:
        checkpoint_filepath = train_config.CKPT_DIR

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=save_weights_only,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1,
    )

    return [tensorboard_callback, model_checkpoint_callback]


"""---------------------------------------------------TRAIN MODEL----------------------------------------------------"""
# Get callbacks.
callbacks = get_callbacks(train_config)
# Define Loss.
loss_fn = partialCE()
loss_2 = tf.keras.losses.Dice(
    reduction='sum_over_batch_size', name='dice'
)
mean_iou = tf.keras.metrics.IoU(
    7,
    id2color.keys())
# Compile model.
model.compile(
    optimizer=tf.keras.optimizers.Adam(train_config.LEARNING_RATE),
    loss=[loss_fn, loss_2],
    metrics=["accuracy", mean_iou],
)

# Train the model, doing validation at the end of each epoch.
history = model.fit(
     train_dataset,
     epochs=train_config.EPOCHS,
     validation_data=valid_dataset,
     callbacks=callbacks
 )



"""--------------------------------------------------EVALUATE MODEL--------------------------------------------------"""
def calculate_per_class_iou(y_true, y_pred, num_classes):
    if len(y_pred.shape) == 4 and y_pred.shape[-1] > 1:
        y_pred = tf.argmax(y_pred, axis=-1)

    if len(y_true.shape) == 4 and y_true.shape[-1] > 1:
        y_true = tf.argmax(y_true, axis=-1)

    # Initialize per-class IoU metric
    mean_iou.update_state(y_true, y_pred)
    confusion = mean_iou.total_cm

    per_class_iou = {}

    for i in range(num_classes):
        true_positive = confusion[i, i]
        false_positive = tf.reduce_sum(confusion[:, i]) - true_positive
        false_negative = tf.reduce_sum(confusion[i, :]) - true_positive

        iou = true_positive / (true_positive + false_positive + false_negative)
        per_class_iou[i] = float(iou.numpy())

    return per_class_iou

def analyze_class_performance(model, test_dataset, num_classes):
    class_names = {i: f"Class {i}" for i in range(num_classes)}
    all_y_true = []
    all_y_pred = []

    for x, y in test_dataset:
        pred = model.predict(x)
        all_y_true.append(y)
        all_y_pred.append(pred)

    y_true = tf.concat(all_y_true, axis=0)
    y_pred = tf.concat(all_y_pred, axis=0)

    # Calculate per-class IoU
    per_class_iou = calculate_per_class_iou(y_true, y_pred, num_classes)

    # Calculate mean IoU
    mean_iou = sum(per_class_iou.values()) / len(per_class_iou)

    # Sort classes by IoU 
    sorted_iou = sorted(per_class_iou.items(), key=lambda x: x[1])

    # Print results
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    for class_idx, iou in sorted_iou:
        class_name = class_names[class_idx]
        print(f"{class_name}: {iou:.4f}")
    return per_class_iou

# if the callback did not work:
# model.save_weights("60.weights.h5")
model.load_weights("60.weights.h5")
evaluate = model.evaluate(valid_dataset)
analyze_class_performance(model, valid_dataset, 7)



def calculate_valid_mean_iou(per_class_iou):
    valid_ious = [iou for iou in per_class_iou.values() if not np.isnan(iou)]
    return sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

per_class_iou = {0: 0.7105, 1: 0.5465, 2: float('nan'),
                3: float('nan'), 4: float('nan'), 5: float('nan'), 6: float('nan')}

valid_mean_iou = calculate_valid_mean_iou(per_class_iou)
print(f"Valid Mean IoU: {valid_mean_iou:.4f}")

#
# # in terminal:
# # tensorboard --logdir best_logs/unlabeled
#
#



"""---------------------------------------------SEMI-SUPERVISED LEARNING---------------------------------------------"""
unlabeled_dir = 'unlabeled'
image_files = [os.path.join(unlabeled_dir, f) for f in os.listdir(unlabeled_dir)]
X_unlabeled = []
for img_path in image_files:
    img = load_img(img_path, target_size=(128, 128)) 
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    X_unlabeled.append(img_array)

X_unlabeled = np.array(X_unlabeled)  # Convert list to numpy array

# Now you can generate predictions
predictions = model.predict(X_unlabeled)
pseudo_labels = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1)


"""-----------------------------------------TEST MODEL AND ENSEMBLE LEARNING-----------------------------------------"""
def inference(models, dataset, samples_to_save):
    num_batches_to_process = 2
    count = 0
    stop_plot = False

    titles = ["Image", "GT Mask", "Pred Mask", "Overlayed Prediction"]

    for idx, data in enumerate(dataset):

        if stop_plot:
            break

        batch_img, batch_mask = data[0], data[1]
        for model in models:
            batch_pred = (model.predict(batch_img)).astype('float32')
        batch_pred = batch_pred.argmax(axis=-1)

        batch_img = batch_img.numpy().astype('uint8')
        # batch_mask = batch_mask.numpy().squeeze(axis=-1)

        for image, mask, pred in zip(batch_img, batch_mask, batch_pred):
            count += 1
            display_image_and_mask([image, mask, pred],
                                   title_list=titles,
                                   figsize=(20, 8),
                                   color_mask=True)
            if count >= samples_to_save:
                stop_plot = True
                break
            plt.savefig(f"samples{count}")


# models = [model] # add another model
# inference(models, test_dataset, samples_to_save=10)
