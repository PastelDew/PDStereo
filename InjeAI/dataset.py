import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as et


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