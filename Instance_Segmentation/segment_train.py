import os
import sys
import argparse
from SketchDataset import SketchDataset

sys.path.append('libs')
from libs.config import Config
import libs.model as modellib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SketchTrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sketchyscene"

    # Batch size is (GPU_COUNT * IMAGES_PER_GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46  # background + 46 classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True

    # image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768

    # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
    TOTAL_EPOCH = 100

    # Optimizer
    # choose from ['sgd', 'adam']
    TRAIN_OPTIMIZER = 'sgd'

    LEARNING_RATE = 0.0001

    # When training, only use the pixels with value 1 in target_mask to contribute to the loss.
    IGNORE_BG = True


def instance_segment_train(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    init_with = kwargs['init_with']
    log_info = kwargs['log_info']

    outputs_base_dir = 'outputs'
    pretrained_model_base_dir = 'pretrained_model'

    save_model_dir = os.path.join(outputs_base_dir, 'snapshot')
    os.makedirs(save_model_dir, exist_ok=True)

    coco_model_path = os.path.join(pretrained_model_base_dir, 'mask_rcnn_coco.pth')
    imagenet_model_path = os.path.join(pretrained_model_base_dir, 'resnet50_imagenet.pth')

    if log_info:
        from tools.logger import Logger
        log_dir = os.path.join(outputs_base_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(log_dir)
    else:
        logger = None

    config = SketchTrainConfig()
    config.display()

    # Training dataset
    dataset_train = SketchDataset(data_base_dir)
    dataset_train.load_sketches("train")
    dataset_train.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(config=config, model_dir=save_model_dir)

    if config.GPU_COUNT:
        model = model.cuda()

    if init_with == "imagenet":
        print("Loading weights from ", imagenet_model_path)
        model.load_weights(imagenet_model_path)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        print("Loading weights from ", coco_model_path)
        model.load_weights(coco_model_path,
                           excludes=["classifier.linear_class.bias", "classifier.linear_class.weight",
                                     "classifier.linear_bbox.bias", "classifier.linear_bbox.weight",
                                     "mask.conv5.bias", "mask.conv5.weight"]
                           )
    elif init_with == "last":
        # Load the last model you trained and continue training
        last_model_path = model.find_last()[1]
        print("Loading weights from ", last_model_path)
        model.load_weights(last_model_path)
    else:
        print("Training from fresh start.")

    # Fine tune all layers
    model.train_model(dataset_train,
                      learning_rate=config.LEARNING_RATE,
                      epochs=config.TOTAL_EPOCH,
                      layers='all',
                      logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_basedir', '-db', type=str, default='../data', help="set the data base dir")
    parser.add_argument('--init_model', '-init', type=str, choices=['imagenet', 'coco', 'last', 'none'],
                        default='none', help="choose a initial pre-trained model")
    parser.add_argument('--log_info', '-log', type=int, choices=[0, 1],
                        default=0, help="Whether log info to tensorboard")
    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_basedir,
        "init_with": args.init_model,
        "log_info": args.log_info
    }

    instance_segment_train(**run_params)