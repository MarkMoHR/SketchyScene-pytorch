# SketchyScene-pytorch

This repository is the official **PyTorch** implementation of semantic/instance segmentation on SketchyScene dataset. (ECCV 2018)

[Tensorflow code](https://github.com/SketchyScene/SketchyScene) | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqing_Zou_SketchyScene_Richly-Annotated_Scene_ECCV_2018_paper.pdf) | [Project Page](https://sketchyscene.github.io/SketchyScene/) | [Dataset](https://github.com/SketchyScene/SketchyScene)

## Outline
- [Semantic Segmentation](#semantic-segmentation)
- [Instance Segmentation](#instance-segmentation)
- [Citation](#citation)

## Semantic Segmentation

See the code under `Semantic_Segmentation` directory.

### Requirements

- Python 3.5
- PyTorch 0.4.1
- torchvision 0.2.1
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
- (Optional) Tensorflow (>= 1.4.0)

### Preparations

- Download the whole [SketchyScene dataset](https://github.com/SketchyScene/SketchyScene) and place them under `data` directory following its instructions.
- Download the ImageNet pre-trained "ResNet-101" pytorch model [here](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtSmdCNDVOVVdINWs) (for initial training) and place it under the `resnet_pretrained_model` directory.
- We provide the implementation of converting our [trained tensorflow model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c) to pytorch model, see [this section](#model-conversion).

### Train

For training based on ImageNet pre-trained "ResNet-101" model:

```
python3 semantic_main.py --mode='train' --init_with='resnet' --log_info=1 --ignore_class_bg=1
```

 - set `--init_with` from [*'resnet'*, *'last'*, *'none'*]. Train from the fresh start with *'none'*. Your lastly trained model will be found automatically if setting *'last'*.
 - `--log_info=1` means log infomation will be summarized and you can check with Tensorboard. **Note** that this requires tensorflow and tensorboard environment. This function benifits from [`logger.py`](https://github.com/MarkMoHR/SketchyScene-pytorch/blob/master/Semantic_Segmentation/tools/logger.py).
 - `--ignore_class_bg=1` means using our proposed background-ignoring strategy. Otherwise, set it to 0.
 - Other training parameters can be modified in [`configs.py`](https://github.com/MarkMoHR/SketchyScene-pytorch/blob/master/Semantic_Segmentation/configs.py).

### Evaluation

Make sure that your trained pytorch model is under the directory `Semantic_Segmentation/outputs/snapshot`. [DenseCRF](https://github.com/lucasb-eyer/pydensecrf) can be used to improve the segmentation performance as a post-processing skill.

For evaluation under *val*/*test* dataset without/with DenseCRF, run:
```
python3 semantic_main.py --mode='val' --dcrf=0
python3 semantic_main.py --mode='test' --dcrf=1
```

You can convert the tensorflow trained model following [this section](#model-conversion) or directly download [here](https://drive.google.com/drive/folders/1em6S6HU0r6_UuWse4n2stbJbAJG8F_Rg?usp=sharing).


### Inference

For inference with the 2-nd image in *val* dataset with DenseCRF, which the background is white, run:

```
python3 semantic_main.py --mode='inference' --infer_dataset='val' --image_id=2 --dcrf=1 --black_bg=0
```

- set `--infer_dataset='test'` for inference under *test* dataset
- set `--image_id` to other number for other image
- set `--black_bg=1` with the result in black background. Otherwise, it is white.

Also, you can try our [converted pytorch model](https://drive.google.com/drive/folders/1em6S6HU0r6_UuWse4n2stbJbAJG8F_Rg?usp=sharing).

### Model Conversion

We provide the implementation of converting our [trained tensorflow model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c) to pytorch model. 

Run the [`convert_tf2pth.py`](https://github.com/MarkMoHR/SketchyScene-pytorch/blob/master/Semantic_Segmentation/tools/convert_tf2pth.py) under `tools` folder like these:

```
python3 convert_tf2pth.py --ignore_class_bg=1 --tf_model_dir=dir/to/tfmodel --display=1 --dataset='val' --image_id=2
python3 convert_tf2pth.py --ignore_class_bg=1 --tf_model_dir==dir/to/tfmodel --display=0
```

 - set `--ignore_class_bg=1` because our tensorflow implementation use this strategy
 - set `--tf_model_dir` to where you place the tensorflow model and checkpoint
 - `--display=1` means a sample scene sketch will be tested and the semantic results from both the tf model and converted pytorch model will be displayed. Remember to set `--dataset` and `--image_id`.
 
We evaluated the converted pytorch model and got the results in the following table:

<table>
  <tr>
    <td rowspan="2"><strong>Model</strong></td>
    <td colspan="2"><strong>OVAcc</strong></td>
    <td colspan="2"><strong>MeanAcc</strong></td>
    <td colspan="2"><strong>MIoU</strong></td>
    <td colspan="2"><strong>FWIoU</strong></td>
  </tr>
  <tr>
    <td>val</td><td>test</td><td>val</td><td>test</td><td>val</td><td>test</td><td>val</td><td>test</td>
  </tr>
  <tr>
    <td><strong>Official TensorFlow model</strong></td> 
    <td>92.94</td> <td>88.38</td> <td>84.95</td> <td>75.92</td> <td>73.49</td> <td>63.10</td> <td>87.10</td> <td>79.76</td>
  </tr>
  <tr>
    <td><strong>Converted Pytorch model</strong></td> 
    <td>92.71</td> <td>88.09</td> <td>84.23</td> <td>75.50</td> <td>71.98</td> <td>62.27</td> <td>86.73</td> <td>79.34</td>
  </tr>
</table>

The results are a bit different mainly due to the Precision Lossing between the two frameworks.


## Instance Segmentation

Coming soon!

## Citation

Please cite the corresponding paper if you found our datasets or code useful:

```
@inproceedings{Zou18SketchyScene,
  author    = {Changqing Zou and
                Qian Yu and
                Ruofei Du and
                Haoran Mo and
                Yi-Zhe Song and
                Tao Xiang and
                Chengying Gao and
                Baoquan Chen and
                Hao Zhang},
  title     = {SketchyScene: Richly-Annotated Scene Sketches},
  booktitle = {ECCV},
  year      = {2018},
  publisher = {Springer International Publishing},
  pages		= {438--454},
  doi		= {10.1007/978-3-030-01267-0_26},
  url		= {https://github.com/SketchyScene/SketchyScene}
}
```

## Credits
- The ResNet-101 pytorch model was converted from caffe model by [ruotianluo](https://github.com/ruotianluo/pytorch-resnet).
- The code for the pytorch DeepLab model is partly borrowed from [chenxi116](https://github.com/chenxi116/pytorch-deeplab).
- The code for tensorboard visualization is from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard).
