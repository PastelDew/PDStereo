"""
Mask R-CNN

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 InjeAI.py train --dataset=/path/to/InjeAI/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 InjeAI.py train --dataset=/path/to/InjeAI/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 InjeAI.py train --dataset=/path/to/InjeAI/dataset --weights=imagenet

    # Apply color splash to an image
    python3 InjeAI.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 InjeAI.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as et

from PDStereo.InjeAI.config import InjeAIConfig
from PDStereo.InjeAI.dataset import InjeAIDataset

import tensorflow as tf
from keras import backend as k

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

def load_class(dataset_dir):
    # annotation file paths which are xml
    annotation_dir = join(dataset_dir, "annotations")
    annotations = [join(dataset_dir, "annotations", f)
                    for f in listdir(annotation_dir)]
    
    # xml tree root list
    annotations = [et.parse(a).getroot() for a in annotations]
    mask_dir = join(dataset_dir, "masks")
    
    classes = []
    for a in annotations:
        for o in a.findall('object'):
            if o.findtext('deleted') != '0':
                continue
            
            mask_path = join(mask_dir, o.find('segm').findtext('mask'))

            if not isfile(mask_path):
                continue
            hasClass = False

            class_name = o.findtext('name')
            if class_name in classes:
                continue
            else:
                classes.append(class_name)
    classes.sort()
    mappedClasses = []
    idx = 1
    for c in classes:
        mappedClasses.append({
            "class": c,
            "id": idx
        })
        idx = idx + 1
    return mappedClasses
    
            
def train(model, classes, layers='all'):
    """Train the model."""
    # Training dataset.
    dataset_train = InjeAIDataset()
    dataset_train.load_InjeAI(args.dataset, "train", model.config.IMAGE_CHANNEL_COUNT, classes)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = InjeAIDataset()
    dataset_val.load_InjeAI(args.dataset, "val", model.config.IMAGE_CHANNEL_COUNT, classes)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCH,
                layers=layers)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB(or -D) image [height, width, channels]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    tempImage = image
    if tempImage.shape[-1] == 4:
        # We ignore the depth channel for visualizing
        tempImage = tempImage[..., :3]
        #tempImage = skimage.color.rgba2rgb(tempImage)
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(tempImage)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, tempImage, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect InjeAIs.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/InjeAI/dataset/",
                        help='Directory of the InjeAI dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--channels', required=False,
                        default=3,
                        type=int,
                        metavar="1, 3 or 4(default: 3)",
                        help='Channels of image to training')
    parser.add_argument('--exclude', required=False,
                        default=False,
                        type=bool,
                        metavar="true or false",
                        help='If you want to use pre-trained weights with different channels then set this option to true.')
    parser.add_argument('--epoch', required=False,
                        default=30,
                        type=int,
                        metavar="The number of epoch(default: 30)",
                        help="The number of epoch")
    parser.add_argument('--steps', required=False,
                        default=100,
                        type=int,
                        metavar="The number of steps per epoch(default: 100)",
                        help="The number of steps per epoch")
    parser.add_argument('--lr', required=False,
                        default=0.001,
                        type=float,
                        metavar="LearningRate(float 0~1)",
                        help="Learning Rate (default: 0.001)")
    parser.add_argument('--layers', required=False,
                        default='all',
                        metavar="Layers for training",
                        help='type layers(all, 4+, ...) for training')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    
    #tf_config = tf.ConfigProto(allow_soft_placement=True,
                #intra_op_parallelism_threads=1,
                #inter_op_parallelism_threads=1,
                #device_count = {'CPU': 1},
                #log_device_placement=False)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 1

    k.tensorflow_backend.set_session(tf.Session(config=tf_config))
    

    # Configurations
    classes = load_class(args.dataset)
    num_classes = len(classes) + 1 # + BG
    if args.command == "train":
        config = InjeAIConfig(args.channels, args.epoch, args.steps, args.lr, num_classes)
    else:
        class InferenceConfig(InjeAIConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig(args.channels, args.epoch, args.steps, args.lr, num_classes)
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"]
        if args.exclude == True:
            exclude.append("conv1")
        model.load_weights(weights_path, by_name=True, exclude=exclude)
    else:
        if args.exclude == True:
            model.load_weights(weights_path, by_name=True, exclude=["conv1"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, classes, args.layers)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
