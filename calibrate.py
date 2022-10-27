# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - config:
    $ python calibrate.py --config 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml'
    
"""

import sys
from pathlib import Path
import os
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
    setup_data_model, get_yolo_predictions, get_calibrator, calibration, calc_mAP, NMS
    )
from utils.torch_utils import  smart_inference_mode

@smart_inference_mode()
def run(
        config=ROOT / 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml',  
        conf_thres=0.001,
        iou_thres_obj=0.60,
        iou_thres_class=0.30,
        where_apply_calib_obj='before_nms', 
        where_apply_calib_class='after_nms',
        calibrator='isotonic',
        size_test=0.6,
        speed_info=False,
        save_dict=True,
        plots=True,
):
    with open(config) as f:
        config = Namespace(**yaml.safe_load(f))
    dataloader, model, device, dt = setup_data_model(config, ROOT)
    num_classes = model.model.nc

    now = datetime.datetime.now()
    save_dir = increment_path(Path(config.project) / config.name / f'calibrate_{str(now.year).zfill(4)}_{str(now.month).zfill(2)}_{str(now.day).zfill(2)}__{str(now.hour).zfill(2)}_{str(now.minute).zfill(2)}', exist_ok=config.exist_ok, mkdir=True)  # increment run
    
    d_ = {
        "FILE": str(FILE),
        "save_dir": str(save_dir),
        "nc": model.model.nc,
        "names": model.names,
        "conf_thres": conf_thres,
        "iou_thres_obj": iou_thres_obj,
        "iou_thres_class": iou_thres_class,
        "where_apply_calib_obj": where_apply_calib_obj,
        "where_apply_calib_class": where_apply_calib_class,
        "calibrator": calibrator,
        "size_test": size_test,
    }
    with open(os.path.join(save_dir, 'var.yaml'), 'w') as file:
        yaml.dump(d_, file)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    dt = Profile(), Profile(), Profile()
    with dt[0]:
        data_dict = get_yolo_predictions(dataloader, model, config, device, dt)
        calib_dict, test_dict = split_dict(data_dict, size_test)
        print("Length calib dict: ", len(calib_dict), " and length test dict: ", len(test_dict))

    calibrator = get_calibrator(calibrator)
    calib_location = "before_nms"
    LOGGER.info(f'\nStarting the calibrations: {calib_location}...')
    calib_obj = (where_apply_calib_obj == calib_location)
    calib_class = (where_apply_calib_class == calib_location)

    if plots:
        os.mkdir(os.path.join(save_dir, "images"))
        plots = os.path.join(save_dir, "images")
    else:
        plots = None

    with dt[1]:
        names_after = calibration(
            ["bbox_xywh", "obj", "class", "bbox_xyxy_scaled"],
            calib_location,
            calib_obj,
            calib_class,
            calib_dict,
            test_dict,
            calibrator,
            conf_thres,
            iou_thres_obj,
            iou_thres_class,
            config,
            num_classes,
            device,
            plots
        )

    print("NMS... test dict attributes: ",test_dict[list(test_dict.keys())[0]].keys())
    print(names_after)
    name_preds = names_after.copy()
    names_after = [names_after[0]+"_nms", names_after[1]+"_nms", names_after[2]+"_nms", names_after[3]+"_nms"]
    NMS(names_after, dataloader, calib_dict, name_preds=name_preds, num_classes=num_classes, opt=config, device=device)
    NMS(names_after, dataloader, test_dict, name_preds=name_preds, num_classes=num_classes, opt=config, device=device)
    names_after[0] = names_after[3]+"_scaled"

    calib_location = "after_nms"
    LOGGER.info(f'\nStarting the calibrations: {calib_location}...')
    calib_obj = (where_apply_calib_obj == calib_location)
    calib_class = (where_apply_calib_class == calib_location)

    with dt[2]:
        names_after = calibration(
            names_after,
            calib_location,
            calib_obj,
            calib_class,
            calib_dict,
            test_dict,
            calibrator,
            conf_thres,
            iou_thres_obj,
            iou_thres_class,
            config,
            num_classes,
            device,
            plots,
        )

    # Print speeds
    if speed_info:
        t = tuple(x.t / len(data_dict) * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms inference, %.1fms before NMS, %.1fms after NMS' % t)

    if save_dict:
        with open(f'{save_dir}/calib_dict.pickle', 'wb') as f:
            pickle.dump(calib_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{save_dir}/test_dict.pickle', 'wb') as f:
            pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Original Results")
    calc_mAP(save_dir, ["bbox_xyxy_scaled_nms", "obj_nms", "class_nms"], device, title="nms", plots=False)

    d_["names_after"] = names_after
    with open(os.path.join(save_dir, 'var2.yaml'), 'w') as file:
        yaml.dump(d_, file)

    print("Calibrated results --> objectness calibration: ",  where_apply_calib_obj, " & class calibration: ", where_apply_calib_class)
    calc_mAP(save_dir, names_after, device, title="calib_nms", plots=False)

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / 'config/opt/opt_Visdrone_detect_S_Hyp0.yaml', help='config_file.yaml path')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou_thres_obj', type=float, default=0.6, help='Objectness IoU threshold')
    parser.add_argument('--iou_thres_class', type=float, default=0.3, help='Class IoU threshold')
    parser.add_argument('--where_apply_calib_obj', type=str, default='before_nms', help='where to apply objectness calibration')
    parser.add_argument('--where_apply_calib_class', type=str, default='after_nms', help='where to apply class calibration')
    parser.add_argument('--calibrator', type=str, default='isotonic', help='which calibrator to use')
    parser.add_argument('--size_test', type=float, default=0.6, help='the split for the dataset')
    parser.add_argument('--speed_info', type=bool, default=False, help='details of the speed')
    parser.add_argument('--save_dict', type=bool, default=True, help='should you save the results')
    parser.add_argument('--plots', type=bool, default=True, help='save the ECE plots')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
