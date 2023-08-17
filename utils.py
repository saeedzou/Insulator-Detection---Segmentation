import os
import random
import shutil
import xml.etree.ElementTree as ET
import numpy as np

def get_unique_dims(path):
    dims = set()
    for img in os.listdir(path):
        tree = ET.parse(os.path.join(path, img))
        root = tree.getroot()
        for size in root.iter('size'):
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
        dims.add((height, width, depth))
    return dims

# make .txt file for each image according to YOLO format
# class x_center y_center width height
# example 0 0.5 0.5 0.5 0.5
def generate_yolo_labels(root, xml_path, txt_path):
    
    for img in os.listdir(xml_path):
        tree = ET.parse(os.path.join(root, xml_path, img))
        root = tree.getroot()
        
        for size in root.iter('size'):
            width = int(size.find('width').text)
            height = int(size.find('height').text)

        for obj in root.iter('object'):
            x_min = int(obj.find('bndbox').find('xmin').text)
            y_min = int(obj.find('bndbox').find('ymin').text)
            x_max = int(obj.find('bndbox').find('xmax').text)
            y_max = int(obj.find('bndbox').find('ymax').text)
            
            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            
            txt_file = img.split('.')[0] + '.txt'
            with open(os.path.join(root, txt_path, txt_file), 'a') as f:
                # save with 6 decimal places
                f.write(f'0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n')

def make_train_valid(root='./InsulatorDataSet', val_size=0.2):
    # make train & valid directories
    os.makedirs(os.path.join(root, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(root, 'valid/images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'valid/labels'), exist_ok=True)
    
    categories = ['Normal_Insulators', 'Defective_Insulators']
    for category in categories:
        images = os.listdir(os.path.join(root, dir, 'images'))
        random.shuffle(images)
        val_images = images[:int(val_size * len(images))]
        train_images = images[int(val_size * len(images)):]
        for image in val_images:
            # copy the image to valid/images
            shutil.copy(os.path.join(root, category, 'images', image), 
                        os.path.join(root, 'valid/images', image))
            # copy the corresponding label to valid/labels
            txt_file =  image.split('.')[0] + '.txt'
            shutil.copy(os.path.join(root, category, 'labels', txt_file), 
                        os.path.join(root, 'valid/labels', txt_file))
        for image in train_images:
            shutil.copy(os.path.join(root, category, 'images', image), 
                        os.path.join(root, 'train/images', image))
            # copy the corresponding label to valid/labels
            txt_file =  image.split('.')[0] + '.txt'
            shutil.copy(os.path.join(root, category, 'labels', txt_file), 
                        os.path.join(root, 'train/labels', txt_file))
    
    # remove categories

# function to convert  normalized x, y, w, h to x1, y1, x2, y2
def convert_xywh_xyxy(bbox, height, width):
    if len(bbox.shape) == 1:
        bbox = np.expand_dims(bbox, axis=0)
    # bbox np array of shape (n, 4)
    x1 = bbox[:, 1] - bbox[:, 3] / 2
    x2 = bbox[:, 1] + bbox[:, 3] / 2
    y1 = bbox[:, 2] - bbox[:, 4] / 2
    y2 = bbox[:, 2] + bbox[:, 4] / 2
    return np.stack([x1 * width, y1 * height, x2 * width, y2 * height], axis=1)

def auto_generate_masks(root, split, sam_predictor, height, width):
    images_path = os.path.join(root, split, 'images')
    labels_path = os.path.join(root, split, 'labels')
    for img in os.listdir(images_path):
        txt_file = os.path.join(labels_path, img.split('.')[0] + '.txt')
        bbox_xywh_norm = np.loadtxt(txt_file)
        bbox_xyxy = convert_xywh_xyxy(bbox_xywh_norm, height=height, width=width)
        labels = np.ones(bbox_xyxy.shape[0]) # all foreground

        results = sam_predictor(source=os.path.join(images_path, img),
                            bboxes=bbox_xyxy,
                            labels=labels)
        # save the mask results in YOLO segmentation format -> class x1 y1 x2 y2 x3 y3 ... xn yn
        masks_xyn = results[0].masks.xyn
        txt_file = os.path.join(labels_path, img.split('.')[0] + '.txt')
        # delete the old txt file
        os.remove(txt_file)
        for i in range(len(masks_xyn)):
            mask = masks_xyn[i]
            mask = np.array(mask)
            mask = mask.reshape(-1)
            mask = mask.tolist()
            mask = [str(x) for x in mask]
            mask = ' '.join(mask)
            mask = '0 ' + mask
            with open(txt_file, 'a') as f:
                f.write(mask + '\n')
