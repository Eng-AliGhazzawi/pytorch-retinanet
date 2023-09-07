import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse

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

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption, font_scale):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 0), 3)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), 3)

def detect_image(image, model, classes, font_scale, save_dir, frame_counter):
    # Make a copy of the original image to draw the captions on
    image_orig = image.copy()

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))


    start_time = time.time()


    with torch.no_grad():
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
            scores, classification, transformed_anchors = model(image.cuda().float())
        else:
            scores, classification, transformed_anchors = model(image.float())

        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[j]
            if score > 0.5:
              caption = '{} {:.3f}'.format(label_name, score)
              cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
              draw_caption(image_orig, (x1, y1-10, x2, y2-10), caption,parser.font)
              print(f"{label_name} (Score: {score:.2f})")

    # Save the modified image with bounding boxes and captions
    img_name = f"frame_{frame_counter:05d}.jpg"
    img_path = os.path.join(save_dir, img_name)
    cv2.imwrite(img_path, image_orig)

    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Processed frame {frame_counter:05d}. Elapsed time: {elapsed_time:.4f} seconds")

    return image_orig


def process_video(video_path, model_path, class_list, save_dir, rate, font_scale, frames):
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    if parser.depth == 18:
        model1 = model.resnet18(num_classes=10)
    elif parser.depth == 34:
        model1 = model.resnet34(num_classes=10)
    elif parser.depth == 50:
        model1 = model.resnet50(num_classes=10)
    elif parser.depth == 101:
        model1 = model.resnet101(num_classes=10)
    elif parser.depth == 152:
        model1 = model.resnet152(num_classes=10)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model1.load_state_dict(torch.load(model_path))
        model1 = model1.cuda()
    else:
        model1.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model1.training = False
    model1.eval()

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = os.path.join(save_dir, "output_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (int(cap.get(3)), int(cap.get(4))))

    frame_counter = 0
    frame_interval = int(frame_rate / rate)  # Calculate the fixed frame interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Check if the current frame should be processed
        if frame_counter % frame_interval == 0:
            processed_image = detect_image(frame, model1, classes, font_scale, save_dir, frame_counter)

            # Write the processed frame and the following frames (replacing the original frames)
            for _ in range(frames):
                out.write(processed_image.astype(np.uint8))
        else:
            # Write the original frame (not processed)
            out.write(frame.astype(np.uint8))

    cap.release()
    out.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for processing a video with object detection.')

    parser.add_argument('--video_path', help='Path to the input video, must be mp4')
    parser.add_argument('--model_path', help='Path to the trained model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--save_dir', help='Path to directory for saving annotated images and processed video')
    parser.add_argument('--rate', help='Frame rate for processing (frames per second)', type=float, default=1.0)
    parser.add_argument('--font', help='Font scale for captions', type=float, default=1.5)
    parser.add_argument('--frames', help='Number of frames to display the detected image', type=int, default=10)
    parser.add_argument('--output_video', help='Name of the output video file', default='output_video.mp4')

    parser = parser.parse_args()

    if not os.path.exists(parser.save_dir):
        os.makedirs(parser.save_dir)

    process_video(parser.video_path, parser.model_path, parser.class_list, parser.save_dir, parser.rate,
                  parser.font, parser.frames)
