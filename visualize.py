import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--save_dir', help='Directory to save the processed images. (Optional)', default="visualization")
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=10)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=10)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=10)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=10)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=10)


    use_gpu = torch.cuda.is_available()

    if use_gpu:
            retinanet.load_state_dict(torch.load(parser.model))
    else:
            retinanet.load_state_dict(torch.load(parser.model, map_location=torch.device('cpu')))
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption_with_score(image, box, caption, score):
        b = np.array(box).astype(int)
        caption_with_score = f"{caption} ({score:.2f})"
        cv2.putText(image, caption_with_score, (b[0], b[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption_with_score, (b[0], b[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                score = scores[idxs[0][j]].item()
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                draw_caption_with_score(img, (x1, y1-10, x2, y2-10), label_name, score)
                print(f"{label_name} (Score: {score:.2f})")

            # Save the processed image with bounding boxes, labels, and scores
            if not os.path.exists(parser.save_dir):
                os.makedirs(parser.save_dir)
            save_dir = parser.save_dir
            save_path = os.path.join(save_dir, f"output_image_{idx}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Image saved at {save_path}")

if __name__ == '__main__':
    main()
