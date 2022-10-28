import yaml


# Change the name yaml file to the correct dataset you will use. 
with open("data/Visdrone.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# If the Python code includes the path or names, make sure to link it back to the variables
names_ = data["names"]
path_ = data["path"]


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