#from mrcnn.config import Config
from mrcnn.config import Config
import numpy as np

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
        assert channel == 3 or channel == 4, "The channel must be 1, 3 or 4! Given: {}".format(channel)
        self.NUM_CLASSES = num_of_classes
        self.IMAGE_CHANNEL_COUNT = channel
        
        if channel == 4:
            self.MEAN_PIXEL = np.append(self.MEAN_PIXEL, 114.8)
        
        self.EPOCH = epoch
        self.STEPS_PER_EPOCH = steps_per_epoch
        self.LEARNING_RATE = learning_rate
        super(InjeAIConfig, self).__init__()
