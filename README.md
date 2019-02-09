# SketchyScene-pytorch

This repository is the official **PyTorch** implementation of semantic segmentation (adapted DeepLab-v2) and instance segmentation (adapted Mask R-CNN) on SketchyScene dataset. (ECCV 2018)

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

See the code under `Instance_Segmentation` directory.

### Requirements

- Python 3.5
- PyTorch 0.4.1
- torchvision 0.2.1
- (Optional) Tensorflow (>= 1.4.0)

### Preparations

- Download the whole [SketchyScene dataset](https://github.com/SketchyScene/SketchyScene) and place them under `data` directory following its instructions.
- Download the coco/imagenet pre-trained model following the instructions under `Instance_Segmentation/pretrained_model`. 
- We provide the implementation of converting our [trained Keras(Tensorflow) model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c) to pytorch model, see [this section](#model-conversion).
- Setup the Non-Maximum Suppression (from [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)) and RoiAlign (from [longcw/RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch)) environment as following:
  ```
  cd libs/nms/src/cuda/
  nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
  cd ../../
  python build.py
  cd ../

  cd roialign/roi_align/src/cuda/
  nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
  cd ../../
  python build.py
  ```
  - choose the value of `--arch` as following:
  
    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | GTX 960M | sm_50 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

### Train
After the preparations, run:

```
python3 segment_train.py
```
or
```
python3 segment_train.py --init_model='coco' --log_info=1
```

- Choose the initial pre-trained model from [*'coco'*, *'imagenet'*, *'last'*] at `--init_model`. Train from the fresh start if not specified. *'last'* denotes your lastly trained model.
- `--log_info=1` means log infomation will be summarized and you can check with Tensorboard. **Note** that this requires tensorflow and tensorboard environment. This function benifits from [`logger.py`](https://github.com/MarkMoHR/SketchyScene-pytorch/blob/master/Semantic_Segmentation/tools/logger.py).
- Other settings can be modified at `SketchTrainConfig` in this file.


### Evaluation

Make sure that your trained model is under the directory `Instance_Segmentation/outputs/snapshot`. 

For evaluation under *val*/*test* dataset, run:
```
python3 segment_evaluate.py --dataset='test' --epochs='0100' --use_edgelist=0
python3 segment_evaluate.py --dataset='val' --epochs='0100' --use_edgelist=1
```

- Set `--epochs` to the last four digits of the name of your trained model.
- Edgelist is used if setting `--use_edgelist=1`. **Note** that if you want to use edgelist as post-processing, make sure you have generated the edgelist labels following the instructions under `Instance_Segmentation/libs/edgelist_utils_matlab`. 

You can convert the keras(tensorflow) trained model following [this section](#model-conversion) or directly download [here](https://drive.google.com/drive/folders/1em6S6HU0r6_UuWse4n2stbJbAJG8F_Rg?usp=sharing).


### Inference

For inference with the 2nd image in *val* dataset without edgelist, run:

```
python3 segment_inference.py --dataset='val' --image_id=2 --epochs='0100' --use_edgelist=0
```

- Inference under *test* dataset if setting `--dataset='test'`
- Try other image if setting `--image_id` to other number
- Set the `--epochs` to the last four digits of your trained model
- Edgelist is used if setting `--use_edgelist=1`. Also make sure the edgelist labels have been generated.

Also, you can try [converted pytorch model](https://drive.google.com/drive/folders/1em6S6HU0r6_UuWse4n2stbJbAJG8F_Rg?usp=sharing).

### Model Conversion

We provide the implementation of converting our [trained keras(tensorflow) model](https://drive.google.com/drive/folders/11sI3IARgAKTf4rut1isQgTOdGKFeyZ1c) to pytorch model. 

Run the [`convert_from_keras.py`](https://github.com/MarkMoHR/SketchyScene-pytorch/blob/master/Instance_Segmentation/tools/convert_from_keras.py) under `tools` folder like this:

```
python3 convert_from_keras.py --keras_model=path/to/keras-model --pytorch_model=path/of/converted-model
```
 
We evaluated the converted pytorch model and got the results in the following table:

<table>
  <tr>
    <td rowspan="2"><strong>Model</strong></td>
    <td colspan="3"><strong>val</strong></td>
    <td colspan="3"><strong>test</strong></td>
  </tr>
  <tr>
    <td>AP</td><td>AP@0.5</td><td>AP@0.75</td><td>AP</td><td>AP@0.5</td><td>AP@0.75</td>
  </tr>
  <tr>
    <td><strong>Official Keras model</strong></td> 
    <td>63.01</td> <td>79.97</td> <td>68.18</td> <td>62.32</td> <td>77.15</td> <td>66.76</td>
  </tr>
  <tr>
     <td><strong>Official Keras model + edgelist</strong></td> 
     <td>63.78</td> <td>80.19</td> <td>68.88</td> <td>63.17</td> <td>77.45</td> <td>67.60</td>
  </tr>
  <tr>
    <td><strong>Converted Pytorch model</strong></td> 
    <td>62.99</td> <td>80.02</td> <td>68.94</td> <td>62.17</td> <td>77.05</td> <td>66.81</td>
  </tr>
  <tr>
     <td><strong>Converted Pytorch model + edgelist</strong></td> 
     <td>63.92</td> <td>80.28</td> <td>69.26</td> <td>63.11</td> <td>77.39</td> <td>67.74</td>
   </tr>
</table>

The results are a bit different mainly due to the Precision Lossing between the two frameworks.

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
- The code for the pytorch Mask R-CNN model is modified from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN), [multimodallearning/pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn), and [jytime/Mask_RCNN_Pytorch](https://github.com/jytime/Mask_RCNN_Pytorch).
- The code for tensorboard visualization is from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard).
