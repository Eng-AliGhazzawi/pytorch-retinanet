# pytorch-retinanet

![img3](https://github.com/yhenon/pytorch-retinanet/blob/master/images/3.jpg)
![img5](https://github.com/yhenon/pytorch-retinanet/blob/master/images/5.jpg)

Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify. Also, it is inspired by Yhenon version at this link (https://github.com/yhenon/pytorch-retinanet).

## Results
Currently, this repo achieves 33.5% mAP at 600px resolution with a Resnet-50 backbone. The published result is 34.0% mAP. The difference is likely due to the use of Adam optimizer instead of SGD with weight decay.

## Installation

1) Make a new environment and clone this repository

2) Install the required packages (Linux only):

```
apt-get install tk-dev python-tk
```

3) Open the repository folder in the environment terminal and perform this  installation command:
	
```
pip install -r requirements.txt

```

## Training

The network can be trained using the `train.py` script. Currently, two dataloaders are available: COCO and CSV. For training on coco, use

```
python train.py --dataset coco --coco_path ../coco --depth 50
```

For training using a custom dataset, with annotations in CSV format (see below), use

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```

Note that the --csv_val argument is optional, in which case no validation will be performed.

## Pre-trained model

A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)

The state dict model can be loaded using: --pre_trained_model <path/to/pre_trained_model.pt>



## Validation

Run `coco_validation.py` to validate the code on the COCO dataset. With the above model, run:

`python coco_validation.py --coco_path ~/path/to/coco --model_path /path/to/model/coco_resnet_50_map_0_335_state_dict.pt`


This produces the following results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
```

For CSV Datasets (more info on those below), run the following script to validate:

`python csv_validation.py --csv_annotations_path path/to/annotations.csv --model_path path/to/model.pt --images_path path/to/images_dir --class_list_path path/to/class_list.csv   (optional) iou_threshold iou_thres (0<iou_thresh<1) `

It produces following resullts:

```
label_1 : (label_1_mAP)
Precision :  ...
Recall:  ...

label_2 : (label_2_mAP)
Precision :  ...
Recall:  ...
```

You can also configure csv_eval.py script to save the precision-recall curve on disk.



## Visualization

To visualize the network detection, use `visualize.py`:

```
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```

To visualize any set of images, use:

```
python visualize_images.py --image_dir <path/to/images> --class_list <path/to/class_list.csv> --model_path <path/to/model.pt> --depth <specify_model_depth> --save_dir <path/to/save/directory> --font <optional>
```

To visualize a video, use:

```
python visualize_video.py --video_path <path/to/video.mp4> --model_path <path/to/model.pt> --class_list <path/to/classes.csv> --save_dir <path/to/output/directory/> --rate <rate_of_proccessing> --depth <model_depth> --frames <number_of_frames_to_display_processed_frames>
```

--rate:

Defines the frequency at which the code processes frames from the video.

For instance, setting it to 1 means the code will process 1 frame every second.

--frames:

Specifies the number of consecutive frames for which a processed frame should be displayed.

Example: If set to 10, every processed frame will be displayed for a duration of 10 frames. This means, if the first 10 frames displayed are derived from the processed first frame, the following frame would be the 11th frame in its original (non-processed) form.

Important Note: If the product of rate and frames exceeds the video's actual frame rate, this will result in an error.


## Model

The retinanet model uses a resnet backbone. You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
image_name.jpg,x1,y1,x2,y2,class_name
```

Note: Every annotation file must be in the same folder as its' related images.

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
image_name.jpg,,,,,
```

A full example:
```
img_001.jpg,837,346,981,456,cow
img_002.jpg,215,312,279,391,cat
img_002.jpg,22,5,89,84,bird
img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.

**Beware**: Some software adds empty lines in place of a labelless image; in such a case, you must follow the instructions in the above example. or it will produce a file not found error.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Acknowledgements

- This repository is essentially an adaptation of [yhenon_implmentation](https://github.com/yhenon/pytorch-retinanet). However, it incorporates several enhancements and additional features, including the capability to visualize images and videos, thus making the tool more user-friendly.

## Examples

![img1](https://github.com/yhenon/pytorch-retinanet/blob/master/images/1.jpg)
![img2](https://github.com/yhenon/pytorch-retinanet/blob/master/images/2.jpg)
![img4](https://github.com/yhenon/pytorch-retinanet/blob/master/images/4.jpg)
![img6](https://github.com/yhenon/pytorch-retinanet/blob/master/images/6.jpg)
![img7](https://github.com/yhenon/pytorch-retinanet/blob/master/images/7.jpg)
![img8](https://github.com/yhenon/pytorch-retinanet/blob/master/images/8.jpg)
