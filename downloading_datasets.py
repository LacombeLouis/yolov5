import yaml

with open("data/coco.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

names_ = data["names"]
path_ = data["path"]

# COCO
from utils.general import download, Path

# Download labels
segments = False  # segment or box labels
dir = Path(path_)  # dataset root dir
url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
download(urls, dir=dir.parent)

# Download data
urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
download(urls, dir=dir / 'images', threads=3)


# # xView
# import json
# import os
# from pathlib import Path

# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# from utils.dataloaders import autosplit
# from utils.general import download, xyxy2xywhn


# def convert_labels(fname=Path('xView/xView_train.geojson')):
#     # Convert xView geoJSON labels to YOLO format
#     path = fname.parent
#     with open(fname) as f:
#         print(f'Loading {fname}...')
#         data = json.load(f)

#     # Make dirs
#     labels = Path(path / 'labels' / 'train')
#     os.system(f'rm -rf {labels}')
#     labels.mkdir(parents=True, exist_ok=True)

#     # xView classes 11-94 to 0-59
#     xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
#                         12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
#                         29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
#                         47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

#     shapes = {}
#     for feature in tqdm(data['features'], desc=f'Converting {fname}'):
#         p = feature['properties']
#         if p['bounds_imcoords']:
#             id = p['image_id']
#             file = path / 'train_images' / id
#             if file.exists():  # 1395.tif missing
#                 try:
#                     box = np.array([int(num) for num in p['bounds_imcoords'].split(",")])
#                     assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
#                     cls = p['type_id']
#                     cls = xview_class2index[int(cls)]  # xView class to 0-60
#                     assert 59 >= cls >= 0, f'incorrect class index {cls}'

#                     # Write YOLO label
#                     if id not in shapes:
#                         shapes[id] = Image.open(file).size
#                     box = xyxy2xywhn(box[None].astype(np.float), w=shapes[id][0], h=shapes[id][1], clip=True)
#                     with open((labels / id).with_suffix('.txt'), 'a') as f:
#                         f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box[0])}\n")  # write label.txt
#                 except Exception as e:
#                     print(f'WARNING: skipping one label for {file}: {e}')

# # Download manually from https://challenge.xviewdataset.org
# dir = Path(yaml['path'])  # dataset root dir
# # urls = ['https://d307kc0mrhucc3.cloudfront.net/train_labels.zip',  # train labels
# #         'https://d307kc0mrhucc3.cloudfront.net/train_images.zip',  # 15G, 847 train images
# #         'https://d307kc0mrhucc3.cloudfront.net/val_images.zip']  # 5G, 282 val images (no labels)
# # download(urls, dir=dir, delete=False)

# # Convert labels
# convert_labels(dir / 'xView_train.geojson')

# # Move images
# images = Path(dir / 'images')
# images.mkdir(parents=True, exist_ok=True)
# Path(dir / 'train_images').rename(dir / 'images' / 'train')
# Path(dir / 'val_images').rename(dir / 'images' / 'val')

# # Split
# autosplit(dir / 'images' / 'train', weights=(0.7, 0.1, 0.2))


# # VisDrone
# from utils.general import download, os, Path
# def visdrone2yolo(dir):
#     from PIL import Image
#     from tqdm import tqdm
#     def convert_box(size, box):
#         # Convert VisDrone box to YOLO xywh box
#         dw = 1. / size[0]
#         dh = 1. / size[1]
#         return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

#     (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
#     pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
#     for f in pbar:
#         img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
#         lines = []
#         with open(f, 'r') as file:  # read annotation.txt
#             for row in [x.split(',') for x in file.read().strip().splitlines()]:
#                 if row[4] == '0':  # VisDrone 'ignored regions' class 0
#                     continue
#                 cls = int(row[5]) - 1
#                 box = convert_box(img_size, tuple(map(int, row[:4])))
#                 lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
#                 with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
#                     fl.writelines(lines)  # write label.txt
# # Download
# dir = Path(path_)  # dataset root dir
# urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
#         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
#         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
#         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
# download(urls, dir=dir, curl=True, threads=4)

# # Convert
# for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
#     visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels


# # VOC
# import xml.etree.ElementTree as ET
# from tqdm import tqdm
# from utils.general import download, Path

# def convert_label(path, lb_path, year, image_id):
#     def convert_box(size, box):
#         dw, dh = 1. / size[0], 1. / size[1]
#         x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
#         return x * dw, y * dh, w * dw, h * dh

#     in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
#     out_file = open(lb_path, 'w')
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)

#     names = list(names_)  # names list
#     for obj in root.iter('object'):
#         cls = obj.find('name').text
#         if cls in names and int(obj.find('difficult').text) != 1:
#             xmlbox = obj.find('bndbox')
#             bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
#             cls_id = names.index(cls)  # class id
#             out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# # # Download
# dir = Path(path_)  # dataset root dir
# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [f'{url}VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
#         f'{url}VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
#         f'{url}VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False, curl=True, threads=3)

# # Convert
# names_ = list(names_.values())
# path = dir / 'images/VOCdevkit'
# for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
#     imgs_path = dir / 'images' / f'{image_set}{year}'
#     lbs_path = dir / 'labels' / f'{image_set}{year}'
#     imgs_path.mkdir(exist_ok=True, parents=True)
#     lbs_path.mkdir(exist_ok=True, parents=True)

#     count=0
#     with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
#         image_ids = f.read().strip().split()
#     for id in tqdm(image_ids, desc=f'{image_set}{year}'):
#         f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
#         lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
#         f.rename(imgs_path / f.name)  # move image
#         convert_label(path, lb_path, year, id)  # convert labels to YOLO format