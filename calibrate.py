# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - config:
    $ python calibrate.py --config 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml'
    
"""

import sys
from pathlib import Path
import os
from sklearn.utils.fixes import config_context
from torch.distributed import pickle
import yaml
import pickle
import datetime
from argparse import Namespace, ArgumentParser

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER, Profile, split_dict, increment_path, print_args
from utils.calibration_utils import (
    setup_data_model, get_yolo_predictions, get_calibrator, calibration
    )
from utils.torch_utils import  smart_inference_mode

@smart_inference_mode()
def run(
        config=ROOT / 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml',  
        conf_thres=0.25,
        iou_thres_obj=0.45,
        iou_thres_class=1000,
        where_apply_calib_obj='before_nms', 
        where_apply_calib_class='after_nms',
        calibrator='isotonic',
        split_calib_test=(0.3, 0.7),
        speed_info=False,
        save_dict=True,
):
    with open(config) as f:
        config = Namespace(**yaml.safe_load(f))
    dataloader, model, device, dt = setup_data_model(config, ROOT)
    num_classes = model.model.nc

    dt = Profile(), Profile(), Profile()
    with dt[0]:
        data_dict = get_yolo_predictions(dataloader, model, config, device, dt)
        calib_dict, test_dict = split_dict(data_dict, split_calib_test)
        print("Length calib dict: ", len(calib_dict), " and length calib dict: ", len(test_dict))

    calibrator = get_calibrator(calibrator)

    calib_location = "before_nms"
    LOGGER.info(f'\nStarting the calibrations: {calib_location}...')
    calib_obj = (where_apply_calib_obj == calib_location)
    calib_class = (where_apply_calib_class == calib_location)

    with dt[1]:
        calibration(
            calib_location,
            calib_obj,
            calib_class,
            calib_dict,
            test_dict,
            calibrator,
            conf_thres,
            iou_thres_obj,
            iou_thres_class,
            dataloader,
            config,
            num_classes,
            device
        )

    calib_location = "after_nms"
    LOGGER.info(f'\nStarting the calibrations: {calib_location}...')
    calib_obj = (where_apply_calib_obj == calib_location)
    calib_class = (where_apply_calib_class == calib_location)

    with dt[2]:
        calibration(
            calib_location,
            calib_obj,
            calib_class,
            calib_dict,
            test_dict,
            calibrator,
            conf_thres,
            iou_thres_obj,
            iou_thres_class,
            dataloader,
            config,
            num_classes,
            device
        )

    # Print speeds
    if speed_info:
        t = tuple(x.t / len(data_dict) * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms inference, %.1fms before NMS, %.1fms after NMS' % t)

    if save_dict:
        now = datetime.datetime.now()
        save_dir = increment_path(Path(config.project) / config.name / f'{str(now.year).zfill(4)}_{str(now.month).zfill(2)}_{str(now.day).zfill(2)}__{str(now.hour).zfill(2)}_{str(now.minute).zfill(2)}__obj_{where_apply_calib_obj}__class_{where_apply_calib_class}', exist_ok=config.exist_ok, mkdir=True)  # increment run
        with open(f'{save_dir}/calib_dict.pickle', 'wb') as f:
            pickle.dump(calib_dict, f)
        with open(f'{save_dir}/test_dict.pickle', 'wb') as f:
            pickle.dump(test_dict, f)



def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml', help='config_file.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres_obj', type=float, default=0.6, help='Objectness IoU threshold')
    parser.add_argument('--iou_thres_class', type=float, default=0.3, help='Class IoU threshold')
    parser.add_argument('--where_apply_calib_obj', type=str, default='before_nms', help='where to apply objectness calibration')
    parser.add_argument('--where_apply_calib_class', type=str, default='after_nms', help='where to apply class calibration')
    parser.add_argument('--calibrator', type=str, default='isotonic', help='which calibrator to use')
    parser.add_argument('--split_calib_test', default=(0.3, 0.7), help='the split for the dataset')
    parser.add_argument('--speed_info', type=bool, default=False, help='details of the speed')
    parser.add_argument('--save_dict', type=bool, default=True, help='should you save the results')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
