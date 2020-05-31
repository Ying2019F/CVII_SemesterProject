"""
Mask R-CNN
Train on the custom tote dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import codecs
import datetime
import numpy as np
import skimage.draw
import cv2 
#For augmentation 
# import imageaug
from imgaug import augmenters as iaa


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

CLASSES = [
    'cookie tin',
    'book',
    'plush duck',
    'toy', 
    'remote',
    'tennis ball', 
    'rubber duck', 
    'heart box',
    'ping pong paddle', 
    'amazon',
    'cat toy'
]

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    #IMAGES_PER_GPU = 4
    IMAGES_PER_GPU = 1
    #IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # Background + Objects Classes

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 70
    #VALIDATION_STEPS = 18
    STEPS_PER_EPOCH = 1
    VALIDATION_STEPS = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # Dont Resize for Inferencing
    # IMAGE_RESIZE_MODE = "pad64"


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        for i in range(1,len(CLASSES)+1):
            self.add_class('object', i, CLASSES[i-1])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        p = os.path.join(dataset_dir, 'images')
        for fname in os.listdir(p):
            image_path = os.path.join(p, fname)
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]
            annotation_filename = os.path.join(
                dataset_dir, 'masks',
                fname.rsplit('.', 1)[0] + '.json'
            )
            with open(annotation_filename, 'r') as f:
                annt_json = json.load(f)

            # Remove image data
            annt_json.pop('imageData', None)

            self.add_image(
                "object",
                image_id=os.path.splitext(fname)[0],  # use file name as a unique image id
                path=image_path,
                annt_json=annt_json,
                width=annt_json['imageWidth'],
                height=annt_json['imageHeight'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        annt_json = info['annt_json']
        mask = np.zeros([info["height"], info["width"], len(annt_json['shapes'])],
                        dtype=np.uint8)
        labels = []
        for i, annt in enumerate(annt_json['shapes']):
            polygon_points = annt['points']
            all_y = [point[1] for point in polygon_points]
            all_x = [point[0] for point in polygon_points]
            rr, cc = skimage.draw.polygon(all_y, all_x)
            mask[rr, cc, i] = 1
            labels.append(CLASSES.index(annt['label'])+1)
            

        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), np.asarray(labels, dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)])#,
        #iaa.Multiply((0.8, 1.5))
    ])

    # augmentation = imageaug.augmenters.Fliplr(0.5)
    print("Training network heads")
    history = model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE*2,
                learning_rate=config.LEARNING_RATE,
                #epochs=1000,
                epochs=100,
                layers='heads',
                augmentation=augmentation)
    #print(history.history)
    #print(history.history['val_loss'])
    # For Hyperoprt
    #last_val_loss = history.history['val_loss'][-1]
    min_val_loss = min(history.history['val_loss'])
    return min_val_loss

def color_splash(image, mask, clss, scores):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    colors = [
        (255,0,0),
        (255,215,0),
         	(124,252,0),
                 	(0,255,255),
                        (127,255,212),
                        (138,43,226),
                         	(160,82,45),
                                (255,0,255),
                                 	(255,140,0),
                                         	(143,188,143),
    ]

    def minmax_wh(poly):
        wmin = hmin = float('+inf')
        wmax = hmax = float('-inf')
        for (w, h) in poly:
            wmin = min(wmin, w)
            wmax = max(wmax, w)
            hmin = min(hmin, h)
            hmax = max(hmax, h)
        return wmax, wmin, hmax, hmin

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask1 = mask.copy()

        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
        print(mask1.shape, image.shape, gray.shape)
        for i, cls in zip(range(mask1.shape[-1]), clss):
            cls = cls - 1
            #color = np.ones(3) * cls / 10 * 255
            color = colors[cls]
            m_i = mask1[:, :, i]
            print(m_i.shape)
            wx, wi, hx, hi = minmax_wh(list(zip(*np.where(m_i))))
            splash[wx, hi:hx] = color
            splash[wi, hi:hx] = color
            splash[wi:wx, hi] = color
            splash[wi:wx, hx] = color
            cls_str = CLASSES[cls] + ' [score={}]'.format(scores[i])
            splash = cv2.putText(
                img=np.copy(splash), text=cls_str,
                org=(hi, wi+6), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255,0,0),
                thickness=5
            )
    else:
        splash = gray.astype(np.uint8)

    #for m in mask:

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
        splash = color_splash(image, r['masks'], r['class_ids'], r['scores'])
        print(r.keys())
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
#  Hyperopt
############################################################
def find_hyperparams(logs):
    from hyperopt import hp
    from hyperopt import fmin, tpe, space_eval

    def create_config(args):
        config = CustomConfig()
        config.LEARNING_RATE = args['LEARNING_RATE']
        config.LEARNING_MOMENTUM = args['LEARNING_MOMENTUM']
        config.WEIGHT_DECAY = args['WEIGHT_DECAY']
        return config

    def objective(args):
        config = create_config(args)
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs)
        history = train(model)
        return history


    space = {
        'LEARNING_RATE': 1 + hp.lognormal('LEARNING_RATE', 1E-5, 1E-2),
        'LEARNING_MOMENTUM': hp.uniform('LEARNING_MOMENTUM', 1E-4, 1),
        'WEIGHT_DECAY': hp.lognormal('WEIGHT_DECAY', 1E-5, 1E-3)
        }

    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
    config = create_config(best)
    return config


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'splash', or 'hyperopt'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
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

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
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
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "hyperopt":
        config = find_hyperparams(args.logs)
        config.display()
        model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
        train(model)        
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
