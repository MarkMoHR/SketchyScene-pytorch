import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import math
import time
from datetime import timedelta
import random
import scipy.io
import argparse

sys.path.append('libs')
sys.path.append('tools')
import adapted_deeplab_pytorch
from configs import Config
from data_loader import load_label
from segment_densecrf import seg_densecrf
from semantic_visualize import visualize_semantic_segmentation

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        self.avg = self.avg * 0.99 + self.val * 0.01


def find_last_model(model_dir, model_fname_start):
    all_models = os.listdir(model_dir)
    max_iter = 0

    for model_name in all_models:
        if model_fname_start not in model_name:
            continue
        model_iter = int(model_name[len(model_fname_start):-4])
        max_iter = max(max_iter, model_iter)

    if max_iter == 0:
        raise Exception("No 'last' trained model is found.")

    last_model_path = os.path.join(model_dir, model_fname_start + str(max_iter) + '.pth')
    return last_model_path, max_iter


def semantic_main(**kwargs):
    mode = str(kwargs['mode'])
    ignore_class_bg = kwargs['ignore_class_bg']

    config = Config()

    print('Mode:', mode, '; ignore_class_bg:', ignore_class_bg)

    snapshot_dir = os.path.join(config.outputs_base_dir, config.snapshot_folder_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    model_fname = os.path.join(snapshot_dir, config.model_fname_start + '%d.pth')

    # BG ignoring
    if ignore_class_bg:
        nSketchClasses = config.nSketchClasses - 1
    else:
        nSketchClasses = config.nSketchClasses
    colorMap = scipy.io.loadmat(os.path.join(config.data_base_dir, 'colorMapC46.mat'))['colorMap']

    use_gpu = torch.cuda.is_available()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # [3, H, W], [0.0-1.0]
    ])

    model = getattr(adapted_deeplab_pytorch, 'resnet101')(num_classes=nSketchClasses, up_mode=config.upsample_mode)

    if mode == 'train':
        init_with = kwargs['init_with']
        log_info = kwargs['log_info']

        model.train()
        start_iter = 0

        # load weight
        if init_with == 'resnet':
            # load weight only in resnet-trained model
            valid_state_dict = model.state_dict()
            trained_state_dict = torch.load(config.resnet_pretrained_model_path)
            for var_name in model.state_dict():
                if var_name in trained_state_dict:
                    valid_state_dict[var_name] = trained_state_dict[var_name]
            model.load_state_dict(valid_state_dict)
            print('Loaded', config.resnet_pretrained_model_path)

        elif init_with == 'last':
            # find the last trained model
            last_model_path, max_iter = find_last_model(snapshot_dir, config.model_fname_start)
            model.load_state_dict(torch.load(last_model_path))
            print('Loaded', last_model_path)
            start_iter = max_iter

        else:  # init_with == 'none'
            print('Training from fresh start.')

        if log_info:
            from logger import Logger
            log_dir = os.path.join(config.outputs_base_dir, config.log_folder_name)
            os.makedirs(log_dir, exist_ok=True)
            logger = Logger(log_dir)

        if use_gpu:
            model = model.cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        opt_param_groups = [{'params': model.conv1.parameters()},
                            {'params': model.bn1.parameters()},
                            {'params': model.layer1.parameters()},
                            {'params': model.layer2.parameters()},
                            {'params': model.layer3.parameters()},
                            {'params': model.layer4.parameters()},
                            {'params': iter([model.fc1_sketch46_c0.weight,
                                             model.fc1_sketch46_c1.weight,
                                             model.fc1_sketch46_c2.weight,
                                             model.fc1_sketch46_c3.weight])},
                            {'params': iter([model.fc1_sketch46_c0.bias,
                                             model.fc1_sketch46_c1.bias,
                                             model.fc1_sketch46_c2.bias,
                                             model.fc1_sketch46_c3.bias]), 'weight_decay': 0.}
                            ]
        if config.upsample_mode == 'deconv':
            opt_param_groups.append({'params': model.deconv.parameters()})

        if config.optimizer == 'sgd':
            optimizer = optim.SGD(opt_param_groups, lr=config.base_lr, momentum=0.9, weight_decay=0.0005)
        elif config.optimizer == 'adam':
            optimizer = optim.Adam(opt_param_groups, lr=config.base_lr, weight_decay=0.0005)
        else:
            raise NameError("Unknown optimizer type %s!" % config.optimizer)

        losses = AverageMeter()
        duration_time_n_step = 0

        for n_iter in range(start_iter, config.max_iteration):
            start_time = time.time()

            # set lr to the same to all layers
            lr = (config.base_lr - config.end_lr) * \
                 math.pow(1 - float(min(n_iter, config.end_decay_step)) / config.end_decay_step, 0.9) + config.end_lr

            if config.multiplied_lr:
                for g in range(6):
                    optimizer.param_groups[g]['lr'] = lr
                optimizer.param_groups[6]['lr'] = lr * 10
                optimizer.param_groups[7]['lr'] = lr * 20
                if config.upsample_mode == 'deconv':
                    optimizer.param_groups[8]['lr'] = lr
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # select image index
            image_idx = random.randint(1, config.nTrainImgs)

            # load images
            image_name = 'L0_sample' + str(image_idx) + '.png'  # e.g. L0_sample5564.png
            # print("Load:", image_name)
            image_path = os.path.join(config.data_base_dir, mode, 'DRAWING_GT', image_name)
            im = datasets.folder.default_loader(image_path)  # [H, W, 3], [0-255]
            inputs = data_transforms(im)  # [3, H, W], ([0.0-1.0]-mean)/std
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            # load gt_label
            label_name = 'sample_' + str(image_idx) + '_class.mat'  # e.g. sample_1_class.mat
            label_path = os.path.join(config.data_base_dir, mode, 'CLASS_GT', label_name)
            gt_label = load_label(label_path)  # shape = [1, H, W], [0, 46]
            if ignore_class_bg:
                gt_label = gt_label - 1  # [-1, 45]
                gt_label[gt_label == -1] = 255  # [0-45, 255]

            h, w = gt_label.shape[1], gt_label.shape[2]
            gt_label = torch.LongTensor(gt_label)  # [1, H, W]
            if use_gpu:
                gt_label = Variable(gt_label.cuda())
            else:
                gt_label = Variable(gt_label)

            outputs = model(inputs.unsqueeze(0))
            if config.upsample_mode == 'deconv':
                outputs_up = outputs
            else:
                outputs_up = nn.UpsamplingBilinear2d((h, w))(outputs)  # [1, nClasses, H, W]

            # calculate loss
            # ignore illegal labels
            gt_label_flatten = gt_label.view(-1, )  # [H * W]
            outputs_up_flatten = torch.t(outputs_up.view(outputs_up.shape[1], -1))  # [H * W, nClasses]

            if ignore_class_bg:  # [0-45, 255], ignore 255
                mask = torch.lt(gt_label_flatten, nSketchClasses)  # lower than 46, [H * W]
                gt_label_flatten = gt_label_flatten[mask]  # [<= H * W]
                outputs_up_flatten = outputs_up_flatten[mask, :]  # [<= H * W, nClasses]

            # outputs_up_flatten: [N, C], gt_label_flatten: [N]
            loss = criterion(outputs_up_flatten, gt_label_flatten)
            losses.update(loss.data[0], 1)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # display loss
            print("Step: {0}. Lr: {1}. Loss: {loss.val:.6f} ({loss.avg:.4f})".format(n_iter + 1, lr, loss=losses))

            # display left time
            duration_time = time.time() - start_time
            duration_time_n_step += duration_time
            if n_iter % config.count_left_time_freq == 0 and n_iter != 0:
                left_step = config.max_iteration - n_iter
                left_sec = left_step / config.count_left_time_freq * duration_time_n_step
                print("Left time: {}".format(str(timedelta(seconds=left_sec))))
                duration_time_n_step = 0

            # summary
            if log_info and n_iter % config.summary_write_freq == 0 and n_iter != 0:
                info = {'loss': loss.item(), 'learning_rate': lr}
                for tag, value in info.items():
                    # print('tag, value', tag, value)
                    logger.scalar_summary(tag, value, n_iter + 1)

            # save model
            if (n_iter + 1) % config.save_model_freq == 0 or (n_iter + 1) >= config.max_iteration:
                torch.save(model.state_dict(), model_fname % (n_iter + 1))
                print('model saved to ' + model_fname % (n_iter + 1))

        print('Training done.')

    elif mode == 'val' or mode == 'test':
        dcrf = kwargs['dcrf']

        def fast_hist(a, b, n):
            """
            :param a: gt
            :param b: pred
            """
            k = (a >= 0) & (a < n)
            return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

        eval_base_dir = os.path.join(config.outputs_base_dir, config.eval_folder_name)
        os.makedirs(eval_base_dir, exist_ok=True)
        model_path, _ = find_last_model(snapshot_dir, config.model_fname_start)

        model.eval()

        model.load_state_dict(torch.load(model_path))
        print('Load', model_path)

        if use_gpu:
            model = model.cuda()

        nImgs = config.nTestImgs if mode == 'test' else config.nValImgs
        outstr = mode + ' mode\n'
        cat_max_len = 16

        hist = np.zeros((config.nSketchClasses, config.nSketchClasses))

        for imgIndex in range(1, nImgs + 1):
            # load images
            image_name = 'L0_sample' + str(imgIndex) + '.png'  # e.g. L0_sample5564.png
            image_path = os.path.join(config.data_base_dir, mode, 'DRAWING_GT', image_name)
            im = datasets.folder.default_loader(image_path)  # [H, W, 3], [0-255]
            im_np = np.array(im, dtype=np.uint8)
            h, w = np.shape(im)[0], np.shape(im)[1]
            inputs = data_transforms(im)  # [3, H, W], [0.0-1.0]
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            # sketchyscene change: we need to scale the inputs to ([0.0-255.0]-mean),
            # following the TF model
            inputs = torch.transpose(torch.transpose(inputs, 0, 2), 0, 1)  # [H, W, 3]
            inputs = inputs * 255.0
            for i in range(3):
                inputs[:, :, i] = inputs[:, :, i] - config.mean[i]
            inputs = torch.transpose(torch.transpose(inputs, 1, 0), 2, 0)  # [3, H, W]

            outputs = model(inputs.unsqueeze(0))
            if config.upsample_mode == 'deconv':
                outputs_up = outputs
            else:
                outputs_up = nn.UpsamplingBilinear2d((h, w))(outputs)  # [1, nClasses, H, W]

            # load gt_label
            label_name = 'sample_' + str(imgIndex) + '_class.mat'  # e.g. sample_1_class.mat
            label_path = os.path.join(config.data_base_dir, mode, 'CLASS_GT', label_name)
            gt_label = load_label(label_path)  # shape = [1, H, W], [0, 46]

            print('#' + str(imgIndex) + '/' + str(nImgs) + ': ' + image_path)

            if dcrf:
                prob_arr = outputs_up.data.cpu().numpy().squeeze(0)  # [nClasses, H, W]
                pred_label = seg_densecrf(prob_arr, im_np, nSketchClasses)  # shape=[H, W], contains [0-45/46]
            else:
                _, pred = torch.max(outputs_up, 1)
                pred_label = pred.data.cpu().numpy()  # [1, H, W]
                pred_label = pred_label.squeeze().astype(np.uint8)  # [H, W], [0-45/46]

            if ignore_class_bg:
                pred_label = pred_label + 1  # [1-46]

            hist += fast_hist(np.squeeze(gt_label).flatten(),
                              pred_label.flatten(),
                              config.nSketchClasses)

        # ignore bg pixel with value 0
        if ignore_class_bg:
            hist = hist[1:, 1:]

        if dcrf:
            print('\nRound', str(nImgs), ', Use CRF')
            outstr += '\nRound: ' + str(nImgs) + ', Use CRF' + '\n'
        else:
            print('\nRound', str(nImgs), ', Not Use CRF')
            outstr += '\nRound: ' + str(nImgs) + ', Not Use CRF' + '\n'

        # overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        print('>>> overall accuracy', acc)
        outstr += '>>> overall accuracy ' + str(acc) + '\n'

        # mAcc
        acc = np.diag(hist) / hist.sum(1)
        mean_acc = np.nanmean(acc)
        print('>>> mean accuracy', mean_acc)
        outstr += '>>> mean accuracy ' + str(mean_acc) + '\n'

        # mIoU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        print('>>> mean IoU', mean_iou)
        outstr += '>>> mean IoU ' + str(mean_iou) + '\n'

        # FWIoU
        freq = hist.sum(1) / hist.sum()
        fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()
        print('>>> freq weighted IoU', fw_iou)
        print('\n')
        outstr += '>>> freq weighted IoU ' + str(fw_iou) + '\n'

        # IoU of each class
        print('>>> IoU of each class')
        outstr += '\n>>> IoU of each class' + '\n'
        for classIdx in range(nSketchClasses):
            if ignore_class_bg:
                cat_name = colorMap[classIdx][0][0]
            else:
                if classIdx == 0:
                    cat_name = 'background'
                else:
                    cat_name = colorMap[classIdx - 1][0][0]

            singlestr = '    >>> '
            cat_len = len(cat_name)
            pad = ''
            for ipad in range(cat_max_len - cat_len):
                pad += ' '
            singlestr += cat_name + pad + str(iou[classIdx])
            print(singlestr)
            outstr += singlestr + '\n'

        # write validation result to txt
        write_path = os.path.join(eval_base_dir, mode + '_results.txt')
        fp = open(write_path, 'a')
        fp.write(outstr)
        fp.close()

    elif mode == 'inference':
        dcrf = kwargs['dcrf']
        image_idx = kwargs['inference_id']
        dataset_type = kwargs['inference_dataset']
        black_bg = kwargs['black_bg']

        infer_result_base_dir = os.path.join(config.outputs_base_dir, config.inference_folder_name, dataset_type)
        os.makedirs(infer_result_base_dir, exist_ok=True)
        model_path, _ = find_last_model(snapshot_dir, config.model_fname_start)

        model.eval()

        model.load_state_dict(torch.load(model_path))
        print('Load', model_path)

        if use_gpu:
            model = model.cuda()

        # load images
        image_name = 'L0_sample' + str(image_idx) + '.png'  # e.g. L0_sample5564.png
        print('image_name', image_name)
        image_path = os.path.join(config.data_base_dir, dataset_type, 'DRAWING_GT', image_name)
        im = datasets.folder.default_loader(image_path)  # [H, W, 3], [0-255]
        im_np = np.array(im, dtype=np.uint8)
        h, w = np.shape(im)[0], np.shape(im)[1]
        inputs = data_transforms(im)  # [3, H, W], [0.0-1.0]
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # sketchyscene change: we need to scale the inputs to ([0.0-255.0]-mean),
        # following the TF model
        inputs = torch.transpose(torch.transpose(inputs, 0, 2), 0, 1)  # [H, W, 3]
        inputs = inputs * 255.0
        for i in range(3):
            inputs[:, :, i] = inputs[:, :, i] - config.mean[i]
        inputs = torch.transpose(torch.transpose(inputs, 1, 0), 2, 0)  # [3, H, W]

        outputs = model(inputs.unsqueeze(0))
        if config.upsample_mode == 'deconv':
            outputs_up = outputs
        else:
            outputs_up = nn.UpsamplingBilinear2d((h, w))(outputs)  # [1, nClasses, H, W]

        if dcrf:
            prob_arr = outputs_up.data.cpu().numpy().squeeze(0)  # [nClasses, H, W]
            pred_label = seg_densecrf(prob_arr, im_np, nSketchClasses)  # shape=[H, W], contains [0-45/46]
        else:
            _, pred = torch.max(outputs_up, 1)
            pred_label = pred.data.cpu().numpy()  # [1, H, W]
            pred_label = pred_label.squeeze().astype(np.uint8)  # [H, W], [0-45/46]

        if ignore_class_bg:
            pred_label = pred_label + 1  # [1-46]

        # filter with binary mask
        pred_label[im_np[:, :, 0] != 0] = 0  # [H, W], [0-46]

        subdir = 'deeplab_output_crf' if dcrf else 'deeplab_output_no_crf'
        save_base_dir = os.path.join(infer_result_base_dir, subdir)
        os.makedirs(save_base_dir, exist_ok=True)
        save_path = os.path.join(save_base_dir, 'sem_result_' + str(image_idx) + '.png')
        visualize_semantic_segmentation(pred_label, colorMap, black_bg=black_bg, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-md', type=str, choices=['train', 'val', 'test', 'inference'],
                        default='train', help="choose a running mode")
    parser.add_argument('--init_with', '-init', type=str, choices=['resnet', 'last', 'none'],
                        default='none', help="initial trained model")
    parser.add_argument('--log_info', '-log', type=int, choices=[0, 1],
                        default=1, help="Whether log info to tensorboard")
    parser.add_argument('--ignore_class_bg', '-ibg', type=int, choices=[0, 1],
                        default=1, help="Whether to ignore background class")

    parser.add_argument('--dcrf', '-crf', type=int, choices=[0, 1],
                        default=1, help="use dense crf or not")
    parser.add_argument('--infer_dataset', '-infd', type=str, choices=['val', 'test'],
                        default='val', help="choose a dataset for inference")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image for inference")
    parser.add_argument('--black_bg', '-bl', type=int, choices=[0, 1],
                        default=0, help="use black or white background for inference")

    args = parser.parse_args()

    if args.image_id == -1 and args.mode == 'inference':
        raise Exception("An image should be chosen for inference.")

    run_params = {
        "mode": args.mode,
        "ignore_class_bg": args.ignore_class_bg == 1,
        "init_with": args.init_with,
        "log_info": args.log_info == 1,
        "dcrf": args.dcrf == 1,
        "inference_id": args.image_id,
        "inference_dataset": args.infer_dataset,
        "black_bg": args.black_bg == 1,
    }

    semantic_main(**run_params)
