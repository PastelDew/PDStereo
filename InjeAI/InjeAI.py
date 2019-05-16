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

############################################################
#  Configurations
############################################################


class InjeAIConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "InjeAI"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + InjeAI

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, channel = 3, epoch = 30, steps_per_epoch = 100, learning_rate = 0.001, num_of_classes = 2):
        assert channel == 1 or channel == 3 or channel == 4, "The channel must be 1, 3 or 4! Given: {}".format(channel)
        self.NUM_CLASSES = num_of_classes
        self.IMAGE_CHANNEL_COUNT = channel
        #if channel == 1 or channel == 3: return
        #elif channel == 4:
        if channel == 4:
            self.MEAN_PIXEL = np.append(self.MEAN_PIXEL, 114.8)
        #elif channel == 1:
            #self.MEAN_PIXEL = [np.sum(self.MEAN_PIXEL) / 3]
        self.EPOCH = epoch
        self.STEPS_PER_EPOCH = steps_per_epoch
        self.LEARNING_RATE = learning_rate
        super()

    ## Custom config by pasteldew
    #BACKBONE = "resnet50"
    #IMAGE_RESIZE_MODE = "crop"
    #IMAGE_MAX_DIM = 256
    #IMAGE_MIN_DIM = 256
    
    #POST_NMS_ROIS_TRAINING = 1000 # default = 1000
    #POST_NMS_ROIS_INFERENCE = 10 # default = 2000
    
    #TRAIN_ROIS_PER_IMAGE = 10 # default = 200
    #MAX_GT_INSTANCES = 50 # default = 100
    
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32) # default = RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    #RPN_TRAIN_ANCHORS_PER_IMAGE = 128 # default = 256


############################################################
#  Dataset
############################################################

class InjeAIDataset(utils.Dataset):
    def load_InjeAI(self, dataset_dir, subset, channels, classes):
        """Load a subset of the InjeAI dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.class_source = "InjeAI"
        self.channel_count = channels
        #self.add_class(self.class_source, 0, "No Masks")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        subset_dir = join(dataset_dir, subset)
        
        imageFiles = [f for f in listdir(subset_dir)
                        if f.lower().endswith('.png') and isfile(join(subset_dir, f))]
        
        # annotation file paths which are xml
        annotations = [join(dataset_dir, "annotations", f.replace(".png", ".xml"))
                        for f in imageFiles]
        
        # xml tree root list
        annotations = [et.parse(a).getroot() for a in annotations]
        mask_dir = join(dataset_dir, "masks")

        class_names = []
        for c in classes:
            class_name, class_id = c["class"], c["id"]
            print("[{}] Class Added: ".format(subset), class_id, class_name)
            class_names.append(class_name)
            self.add_class(self.class_source, class_id, class_name)
        
        for a in annotations:
            filename = a.findtext('filename').replace('.jpg', '.png')
            image_path = join(subset_dir, filename)
            image = skimage.io.imread(image_path)
            if len(image.shape) <= 1:
                continue
                
            masks = []
            for o in a.findall('object'):
                if o.findtext('deleted') != '0':
                    continue
                
                mask_path = join(mask_dir, o.find('segm').findtext('mask'))

                if not isfile(mask_path):
                    continue
                hasClass = False

                try:
                    class_name = o.findtext('name')
                    class_id = classes[class_names.index(class_name)]['id']

                    masks.append({"path": mask_path, "class_id": class_id})
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    continue
            
            height, width = image.shape[:2]
            self.add_image(
                self.class_source,
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                masks=masks)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim == 1:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4 and self.channel_count == 3:
            image = image[..., :3]
        elif image.shape[-1] == 3 and self.channel_count == 4:
            alpha = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
            image = np.dstack((image, alpha))
        elif image.shape[-1] == 4 and self.channel_count == 4:
            image[..., 3] = skimage.exposure.equalize_hist(image[..., 3])
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a InjeAI dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != self.class_source or len(info["masks"]) == 0:
            print("Passing load_mask to default function.", info["source"], len(info["masks"]))
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["masks"])],
                        dtype=np.uint8)
        class_ids = np.zeros([mask.shape[-1]], dtype=np.int32)
        for i, m in enumerate(info["masks"]):
            mask_path = m['path']
            class_id = m['class_id']

            mask_image = skimage.io.imread(mask_path)
            if mask_image.shape[-1] == 4:
                mask_image = mask_image[..., :3]
                mask_image = skimage.color.rgb2gray(mask_image)
            elif mask_image.shape[-1] == 3:
                mask_image = skimage.color.rgb2gray(mask_image)
            mask_image = mask_image > 0
            mask[:, :, i] = mask_image
            class_ids[i] = class_id

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "InjeAI":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def load_Class(dataset_dir):
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
    
            
def train(model, classes):
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
                layers='all')


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
    classes = load_Class(args.dataset)
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
        train(model, classes)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
