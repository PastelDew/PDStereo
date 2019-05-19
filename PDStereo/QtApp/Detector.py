from PDStereo.InjeAI import InjeAI
from PDStereo.InjeAI.config import InjeAIConfig
from PDStereo.InjeAI.dataset import InjeAIDataset

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import cv2
import colorsys
import random
from skimage.measure import find_contours
from os.path import join
from matplotlib.patches import Polygon
import numpy as np

class Detector():
    def __init__(self, classes, device="/cpu:0", root_dir="../../DL/"):
        assert type(classes) is list
        assert len(classes) > 0
        assert type(classes[0]) is dict
        assert type(classes[0]['class']) is not None
        assert type(classes[0]['id']) is not None

        self.weights = None
        self.ROOT_DIR = root_dir
        self.MODEL_DIR = join(self.ROOT_DIR, 'logs/')
        self.channel = 3

        class InferenceConfig(InjeAIConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.classes = classes
        self.class_names = [c['class'] for c in classes]
        self.colors = self.random_colors(len(self.class_names))
        self.class_names.sort()

        colors = {}
        for i in range(len(self.class_names)):
            colors[self.class_names[i]] = self.colors[i]
        self.colors = colors

        self.class_names.insert(0, 'BG')
        self.num_of_classes = len(classes) + 1 # + BG
        self.inference_config = InferenceConfig
        self.config = None
        self.model = None
        self.isDetecting = False
        
    def load_weights(self, path, channel):
        self.channel = channel
        self.config = self.inference_config(channel=channel, num_of_classes=self.num_of_classes)
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR,
                            config=self.config)
        self.model.load_weights(path, by_name=True)
        self.model.keras_model._make_predict_function()

    def detect(self, images):
        self.isDetecting = True
        #results = self.model.detect(images, verbose=1)
        results = self.model.detect(images)
        self.isDetecting = False
        return results

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def get_instances_image(self, image, boxes, masks, class_ids,
                      scores=None,
                      show_mask=True, show_bbox=True,
                      captions=None):
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Show area outside image boundaries.
        height, width = image.shape[:2]

        masked_image = np.asarray(image.copy(), dtype=np.uint8)
        for i in range(N):
            class_id = class_ids[i]
            label = self.class_names[class_id]
            color = self.colors[label]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                color_for_cv = tuple([int(255 * c) for c in color])
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), color_for_cv, 2)
                #cv2.rectangle(masked_image, (x1, y1), (x2 - x1, y2 - y1), color_for_cv, 2)

            # Label
            if not captions:
                score = scores[i] if scores is not None else None
                x = random.randint(x1, (x1 + x2) // 2)
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            cv2.putText(masked_image, caption, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

            #masked_image = masked_image.astype(np.uint32)

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            """
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
            """
        return masked_image.astype(np.uint8)
