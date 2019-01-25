import argparse
import numpy as np
import scipy.io
import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append('..')
sys.path.append('../libs')
import adapted_deeplab_pytorch
import adapted_deeplab_tf
from data_loader import load_image
from configs import Config
from semantic_visualize import visualize_semantic_segmentation


class ConvertSemanticTF2Pth(object):
    def __init__(self, ignore_class_bg, display, disp_img_id, disp_dataset, tf_model_dir):
        self.ignore_class_bg = ignore_class_bg
        self.display = display
        self.disp_img_id = disp_img_id
        self.disp_dataset = disp_dataset
        self.tf_model_dir = tf_model_dir

        self.config = Config()
        # BG ignoring
        if self.ignore_class_bg:
            self.nSketchClasses = self.config.nSketchClasses - 1
        else:
            self.nSketchClasses = self.config.nSketchClasses

        self.upsample_mode = self.config.upsample_mode
        self.sketchyscene_data_base = os.path.join('..', self.config.data_base_dir)

        self.image_name = 'L0_sample' + str(self.disp_img_id) + '.png'  # e.g. L0_sample5564.png
        self.image_path = os.path.join(self.sketchyscene_data_base, self.disp_dataset, 'DRAWING_GT', self.image_name)
        self.colorMap = scipy.io.loadmat(os.path.join(self.sketchyscene_data_base, 'colorMapC46.mat'))['colorMap']

        self.tf_start_name = 'ResNet/'
        self.tf_weight_dict = {}
        self.read_tf_weight()

        self.pth_model_path = os.path.join(self.tf_model_dir, 'sketchyscene_deeplab101_epoch100000.pth')
        self.write_pytorch_weight()

    def read_tf_weight(self):
        model = adapted_deeplab_tf.DeepLab(num_classes=self.nSketchClasses,
                                           upsample_mode=self.upsample_mode,
                                           ignore_class_bg=self.ignore_class_bg)
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(self.tf_model_dir)
        snapshot_loader = tf.train.Saver()
        print('Trained model found, loaded', ckpt.model_checkpoint_path)
        snapshot_loader.restore(sess, ckpt.model_checkpoint_path)

        weight_keys = [var.name[:-2] for var in tf.global_variables()]
        weight_vals = sess.run(tf.global_variables())
        for weight_key, weight_val in zip(weight_keys, weight_vals):
            if 'factor' not in weight_key and 'Adam' not in weight_key and 'beta1_power' not in weight_key \
                    and 'beta2_power' not in weight_key and 'global_step' not in weight_key:
                # print(weight_key, np.array(weight_val).shape)
                self.tf_weight_dict[weight_key] = weight_val

        print('Obtain tf_weight_dict done!')

        # get output
        if self.display:
            infer_image, infer_image_raw = load_image(self.image_path, self.config.mean, True)  # shape = [1, H, W, 3]
            feed_dict = {model.images: infer_image, model.labels: 0}
            self.pred_label_tf = sess.run([model.pred_label], feed_dict=feed_dict)[0]  # [1, H, W, 1]

            if self.ignore_class_bg:
                pred_label_tf = self.pred_label_tf + 1
            else:
                pred_label_tf = self.pred_label_tf

            pred_label_tf = np.squeeze(pred_label_tf)
            pred_label_tf[infer_image_raw[:, :, 0] != 0] = 0  # [H, W]
            visualize_semantic_segmentation(pred_label_tf, self.colorMap)

    def write_pytorch_weight(self):
        model = getattr(adapted_deeplab_pytorch, 'resnet101')(num_classes=self.nSketchClasses,
                                                              up_mode=self.upsample_mode)
        model.eval()

        self.used_keys = []

        # tf-group1
        self.convert_conv(model.conv1, self.compl('group_1/conv1/DW'))
        self.convert_bn(model.bn1, self.compl('group_1/bn_conv1/gamma'), self.compl('group_1/bn_conv1/beta'),
                        self.compl('group_1/bn_conv1/mean'), self.compl('group_1/bn_conv1/variance'))

        self.num_unit = [3, 4, 23, 3]
        # tf-group2 / pth-layer1 to tf-group5 / pth-layer4
        self.convert_layer(model.layer1, 'group_2', self.num_unit[0])
        self.convert_layer(model.layer2, 'group_3', self.num_unit[1])
        self.convert_layer(model.layer3, 'group_4', self.num_unit[2])
        self.convert_layer(model.layer4, 'group_5', self.num_unit[3])

        # fc layers
        self.convert_fc(model.fc1_sketch46_c0,
                        self.compl('fc_final_sketch46/conv0/DW'), self.compl('fc_final_sketch46/conv0/biases'))
        self.convert_fc(model.fc1_sketch46_c1,
                        self.compl('fc_final_sketch46/conv1/DW'), self.compl('fc_final_sketch46/conv1/biases'))
        self.convert_fc(model.fc1_sketch46_c2,
                        self.compl('fc_final_sketch46/conv2/DW'), self.compl('fc_final_sketch46/conv2/biases'))
        self.convert_fc(model.fc1_sketch46_c3,
                        self.compl('fc_final_sketch46/conv3/DW'), self.compl('fc_final_sketch46/conv3/biases'))

        # upsample layer
        self.convert_up(model.deconv, self.compl('fc_final_sketch46/W_up'), self.compl('fc_final_sketch46/b_up'))

        print('Conversion complete!')

        # checking
        print('Checking: whether all TF variables are used...')
        assert len(self.tf_weight_dict) == len(self.used_keys)
        print('Pass!')

        # display semantic results
        if self.display:
            use_gpu = torch.cuda.is_available()
            data_transforms = transforms.Compose([
                transforms.ToTensor(),  # [3, H, W], [0.0-1.0]
            ])

            model = model.cuda()
            im = datasets.folder.default_loader(self.image_path)  # [H, W, 3], [0-255]
            im_np = np.array(im, dtype=np.uint8)
            inputs = data_transforms(im)  # [3, H, W], ([0.0-1.0]-mean)/std
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            # sketchyscene change: we need to scale the inputs to ([0.0-255.0]-mean),
            # following the TF model
            inputs = torch.transpose(torch.transpose(inputs, 0, 2), 0, 1)  # [H, W, 3]
            inputs = inputs * 255.0
            for i in range(3):
                inputs[:, :, i] = inputs[:, :, i] - self.config.mean[i]
            inputs = torch.transpose(torch.transpose(inputs, 1, 0), 2, 0)  # [3, H, W]

            outputs = model(inputs.unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred_label_pth = pred.data.cpu().numpy()  # [1, H, W]
            pred_label_pth = pred_label_pth.squeeze().astype(np.uint8)  # [H, W], [0-45/46]

            if self.ignore_class_bg:
                pred_label_pth = pred_label_pth + 1  # [1-46]

            pred_label_pth[im_np[:, :, 0] != 0] = 0  # [H, W], [0-46]
            visualize_semantic_segmentation(pred_label_pth, self.colorMap)

        # save converted model
        torch.save(model.state_dict(), self.pth_model_path)
        print('PyTorch model saved to {}'.format(self.pth_model_path))

    def convert_layer(self, layer, sub_prefix, units):
        assert isinstance(layer, nn.Sequential)
        for unit in range(units):
            tf_grp_start = sub_prefix + '_' + str(unit)  # group_2_0 / layer1.0
            self.convert_conv(layer[unit].conv1, self.compl(tf_grp_start + '/block_1/conv/DW'))
            self.convert_bn(layer[unit].bn1,
                            self.compl(tf_grp_start + '/block_1/bn/gamma'),
                            self.compl(tf_grp_start + '/block_1/bn/beta'),
                            self.compl(tf_grp_start + '/block_1/bn/mean'),
                            self.compl(tf_grp_start + '/block_1/bn/variance'))

            self.convert_conv(layer[unit].conv2, self.compl(tf_grp_start + '/block_2/conv/DW'))
            self.convert_bn(layer[unit].bn2,
                            self.compl(tf_grp_start + '/block_2/bn/gamma'),
                            self.compl(tf_grp_start + '/block_2/bn/beta'),
                            self.compl(tf_grp_start + '/block_2/bn/mean'),
                            self.compl(tf_grp_start + '/block_2/bn/variance'))

            self.convert_conv(layer[unit].conv3, self.compl(tf_grp_start + '/block_3/conv/DW'))
            self.convert_bn(layer[unit].bn3,
                            self.compl(tf_grp_start + '/block_3/bn/gamma'),
                            self.compl(tf_grp_start + '/block_3/bn/beta'),
                            self.compl(tf_grp_start + '/block_3/bn/mean'),
                            self.compl(tf_grp_start + '/block_3/bn/variance'))

            if unit == 0:
                assert isinstance(layer[unit].downsample, nn.Sequential)
                self.convert_conv(layer[unit].downsample[0], self.compl(tf_grp_start + '/block_add/conv/DW'))
                self.convert_bn(layer[unit].downsample[1],
                                self.compl(tf_grp_start + '/block_add/bn/gamma'),
                                self.compl(tf_grp_start + '/block_add/bn/beta'),
                                self.compl(tf_grp_start + '/block_add/bn/mean'),
                                self.compl(tf_grp_start + '/block_add/bn/variance'))

    def convert_conv(self, conv2d, weights_key):
        weights = self.tf_weight_dict[weights_key]

        # TF: [kernel_size, kernel_size, in_channels, out_channels]
        # PyTorch: [out_channels, in_channels, kernel_size, kernel_size]
        weights = np.transpose(weights, (3, 2, 0, 1))
        assert conv2d.weight.shape == self.Param(weights).shape, '{0} vs {1}'.format(conv2d.weight.shape,
                                                                                     self.Param(weights).shape)
        conv2d.weight = self.Param(weights)
        self.used_keys += [weights_key]

    def convert_bn(self, bn, gamma_key, beta_key, moving_mean_key, moving_var_key):
        gamma = self.tf_weight_dict[gamma_key]
        beta = self.tf_weight_dict[beta_key]
        moving_mean = self.tf_weight_dict[moving_mean_key]
        moving_var = self.tf_weight_dict[moving_var_key]
        assert bn.weight.shape == self.Param(gamma).shape
        assert bn.bias.shape == self.Param(beta).shape
        assert bn.running_mean.shape == self.Tensor(moving_mean).shape
        assert bn.running_var.shape == self.Tensor(moving_var).shape

        bn.weight = self.Param(gamma)
        bn.bias = self.Param(beta)
        bn.running_mean = self.Tensor(moving_mean)
        bn.running_var = self.Tensor(moving_var)
        self.used_keys += [gamma_key, beta_key, moving_mean_key, moving_var_key]

    def convert_fc(self, fc, weights_key, biases_key):
        weights = self.tf_weight_dict[weights_key]
        biases = self.tf_weight_dict[biases_key]

        # TF: [kernel_size, kernel_size, in_channels, out_channels]
        # PyTorch: [out_channels, in_channels, kernel_size, kernel_size]
        weights = np.transpose(weights, (3, 2, 0, 1))
        assert fc.weight.shape == self.Param(weights).shape
        assert fc.bias.shape == self.Param(biases).shape

        fc.weight = self.Param(weights)
        fc.bias = self.Param(biases)
        self.used_keys += [weights_key, biases_key]

    def convert_up(self, up, weights_key, biases_key):
        weights = self.tf_weight_dict[weights_key]
        biases = self.tf_weight_dict[biases_key]

        # TF: [kernel_size, kernel_size, in_channels, out_channels]
        # PyTorch: [out_channels, in_channels, kernel_size, kernel_size]
        weights = np.transpose(weights, (3, 2, 0, 1))
        assert up.weight.shape == self.Param(weights).shape
        assert up.bias.shape == self.Param(biases).shape

        up.weight = self.Param(weights)
        up.bias = self.Param(biases)
        self.used_keys += [weights_key, biases_key]

    def compl(self, key):
        return self.tf_start_name + key

    def Param(self, x):
        return torch.nn.Parameter(torch.from_numpy(x))

    def Tensor(self, x):
        return torch.from_numpy(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore_class_bg', '-ibg', type=int, choices=[0, 1],
                        default=1, help="Whether to ignore background class")
    parser.add_argument('--display', '-dsp', type=int, choices=[0, 1],
                        default=1, help="Whether to display results from two models")
    parser.add_argument('--dataset', '-ds', type=str, choices=['val', 'test'],
                        default='val', help="choose a dataset for display")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image for display")
    parser.add_argument('--tf_model_dir', '-tf', type=str, default='../outputs/snapshot/TF', help="the dir of tf model")

    args = parser.parse_args()

    if args.display == 1 and args.image_id == -1:
        raise Exception("An image should be chosen for display.")

    ConvertSemanticTF2Pth(ignore_class_bg=args.ignore_class_bg == 1,
                          display=args.display == 1,
                          disp_dataset=args.dataset,
                          disp_img_id=args.image_id,
                          tf_model_dir=args.tf_model_dir)
