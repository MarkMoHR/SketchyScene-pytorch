import numpy as np
import math

class Config(object):
    ############################################
    #           dataset params
    ############################################

    # The number of sketch classes
    nSketchClasses = 47

    # The number of training sketches
    nTrainImgs = 5617

    # The number of val sketches
    nValImgs = 535

    # The number of test sketches
    nTestImgs = 1113

    ############################################
    #      common params
    ############################################

    # The mean array
    mean = (104.00698793, 116.66876762, 122.67891434)

    # The base folder of outputs
    outputs_base_dir = 'outputs'

    # The folder of trained model
    snapshot_folder_name = 'snapshot'

    # The folder of train_images
    data_base_dir = '../data'

    ############################################
    #   sketch segmentation training parameter
    ############################################

    # The pre_trained resnet model path for training
    resnet_pretrained_model_path = 'resnet_pretrained_model/resnet101.pth'

    # The log dir during training
    log_folder_name = 'log'

    # The start name of model_fname
    model_fname_start = 'sketchyscene_deeplab101_epoch'

    # learning rate
    base_lr = 0.0001

    # end learning rate
    end_lr = 0.00001

    # The max_iteration of training
    max_iteration = 100000

    # The ending iteration of lr decay
    end_decay_step = 70000

    # The upsample mode of resizing to image_size. Choose from [bilinear, deconv]
    upsample_mode = 'deconv'

    # The optimizer used to train. Choose from [sgd, adam]
    optimizer = 'adam'

    # Whether add multiplied lr to fc layers
    multiplied_lr = True

    # Write summary frequence
    summary_write_freq = 50

    # Save model frequence
    save_model_freq = 20000

    # Count left time frequence
    count_left_time_freq = 100

    ############################################
    #    eval & inference parameter
    ############################################

    # The folder of eval results
    eval_folder_name = 'eval_results'

    # The folder of inference results
    inference_folder_name = 'inference_results'

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


if __name__ == "__main__":
    config = Config()
    config.display()
