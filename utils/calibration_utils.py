import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrics import box_iou
from matplotlib.offsetbox import TextArea, AnnotationBbox

from argparse import Namespace
import yaml
from pathlib import Path
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
import pickle
import torch

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_dataset, xywh2xyxy, Profile, check_img_size, colorstr, scale_boxes, non_max_suppression)
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
from utils.metrics import ap_per_class, ConfusionMatrix
from val import process_batch

def get_data_pred(names, preds_, chosen_dict, image_path, num_classes):
    """
    This function uses the predictions and saves them in the dictionnary passed.

    Parameters
    ----------
    preds_ : _type_
        Predictions made by the Yolov5 algorithm. Need to have in the last five 
        indices the objectness confidence, the class confidences and the xyxy scaled boxed.
    chosen_dict : _type_
        The dictionnary on which to save the observations.
    image_path : _type_
        The path of the image, the key where you want to save in the dictionnary.
    name : _type_
        The name that you want to give to this specific prediction.
    """
    if isinstance(image_path, list) or isinstance(image_path, tuple):
        image_path = image_path[0]
    chosen_dict[image_path][names[0]] = preds_[:, :4].clone().cpu().numpy()
    chosen_dict[image_path][names[1]] = preds_[:, -num_classes-1:-num_classes].clone().cpu().numpy()
    chosen_dict[image_path][names[2]] = preds_[:, -num_classes:].clone().cpu().numpy()

# Put the predictions in a dictionnary (preds before and after NMS)
def setup_data_model(opt, ROOT):
    # Directories
    data = ROOT / opt.data  # dataset.yaml path
    with open(data) as f:
        opt_data = Namespace(**yaml.safe_load(f))
    # (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    weights = ROOT / opt.weights  # model path or triton URL
    device = select_device(opt.device, batch_size=opt.batch_size)
    model = DetectMultiBackend(weights, device=device, dnn=opt.dnn, data=data, fp16=opt.half)
    opt.half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if opt.half else model.float()
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    ncm = model.model.nc
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images

    # Data
    data = check_dataset(data)  # check
    nc = 1 if opt.single_cls else int(opt_data.nc)  # number of classes
    # names = {0: 'item'} if opt.single_cls and len(opt_data.names) != 1 else opt_data.names  # class names
    # names = model.names if hasattr(model, 'names') else model.module.names  # get class names

    # Other
    dt = Profile(), Profile(), Profile()  # profiling times
    assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                        f'classes). Pass correct combination of --weights and --data that are trained together.'

    model.warmup(imgsz=(1 if pt else opt.batch_size, 3, imgsz, imgsz))  # warmup
    dataloader = create_dataloader(data[task],
                                    imgsz,
                                    opt.batch_size,
                                    stride,
                                    opt.single_cls,
                                    workers=opt.workers,
                                    prefix=colorstr(f'{task}: '))[0]
                                
    return dataloader, model, device, dt

def get_yolo_predictions(dataloader, model, opt, device, dt):
    # count = 0
    data_dict = {}
    nc = model.model.nc
    pbar = tqdm(dataloader, desc=('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95'), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for im, targets, paths, shapes in pbar:
        with dt[0]:
            if device.type != 'cpu':
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if opt.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im)

        preds_ = preds.clone()

        # Adjusting the size of the image for height and width 
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling

        # NMS
        with dt[2]:
            preds_nms = non_max_suppression(
                preds,
                opt.conf_thres_nms,
                opt.iou_thres_nms,
                labels=lb,
                multi_label=opt.multi_label_nms,
                agnostic=opt.agnostic_nms,
                max_det=opt.max_det_nms,
                output_confs=True,
                )

        # Initiating dictionnary for this picture
        data_dict[paths[0]] = {}

        for si, pred_ in enumerate(preds_):        
            # Outputs of model are in xywh
            # Saving predicitions as xyxy and outputs of model are in xywh
            pred_[:, :4] = xywh2xyxy(pred_[:, :4])
            scale_boxes(im[si].shape[1:], pred_[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            # Targets are in xywh
            tbox = xywh2xyxy(targets[:, 2:].clone())
            scale_boxes(im[si].shape[1:], tbox[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
        # Normal predictions
        data_dict[paths[0]]['bbox_xywh'] = preds[0][:, :4].clone().cpu().numpy()
        get_data_pred(["bbox_xyxy_scaled", "obj", "class"], preds_[0], data_dict, paths, num_classes=nc)

        for si, pred_nms_ in enumerate(preds_nms):
            scale_boxes(im[si].shape[1:], pred_nms_[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Normal predictions after NMS
        get_data_pred(["bbox_xyxy_scaled_nms", "obj_nms", "class_nms"], preds_nms[0], data_dict, paths, num_classes=nc)

        # True images
        data_dict[paths[0]]['annot_image_shape'] = (height, width)
        data_dict[paths[0]]['annot_bbox_xyxy_scaled'] = tbox.clone().cpu().numpy()
        data_dict[paths[0]]['annot_bbox'] = targets[:, 2:].clone().cpu().numpy()
        data_dict[paths[0]]['annot_class'] = targets[:, 1:2].clone().cpu().numpy()

        # if count > 20:
        #     break
        # count+=1
    return data_dict


# Create dictionnary with the new labels, the y_true
def calib_prep(names, dict_, num_classes, device, conf_thres, iou_thres_obj, iou_thres_class, calib_location, obj_calib=True, class_calib=False):
    """
    Preparing for calibration, thereby setting the "true" y to calibrate against.

    Parameters
    ----------
    true_dict : _type_
        Dictionnary of all the initial predictions, where the true values are saved.
    chosen_dict : _type_
        Dictionnary where the predictions you want to calibrate are saved.
    chosen_dict_new : _type_
        Dictionnary where you want to save the new calibrated values and where the "true" y's are saved.
    conf_thres : int, optional
        The minimum objectness confidence threshold for which bboxs will be considered, by default 0.02
    iou_thres : int, optional
        The minimum IoU threshold with a true annotated bbox for which bboxs will be considered, by default 0
    obj_calib : bool, optional
        Boolean to determine if you want to to calibrate objectness, by default True
    class_calib : bool, optional
        Boolean to determine if you want to to calibrate classes, by default False
    where_apply : str, optional
        Determines the name of the observations to use when calibrating, by default "pred"
    num_classes : int, optional
        The number of classes that can be predicted in the dataset, by default 4
    """
    names_copy = names.copy()
    if obj_calib is True and class_calib is True:
        add_name = "_OCcalib"
    elif obj_calib is True:
        add_name = "_Ocalib"
    else:
        add_name = "_Ccalib"

    names = [names[0]+add_name, names[1]+add_name, names[2]+add_name, names[3]+add_name]
    for i, image_path in enumerate(tqdm(dict_)):
        objectness_idx = np.where(dict_[image_path][names_copy[1]] > conf_thres)[0]
        num_obj_ = len(objectness_idx)

        # If no bboxs have the matching criterion, then forget this image
        if num_obj_==0:
            dict_[image_path]["has_detec"] = 0
        else:
            dict_[image_path]["has_detec"] = 1
            dict_[image_path][names[0]] = dict_[image_path][names_copy[0]][objectness_idx]
            dict_[image_path][names[1]] = dict_[image_path][names_copy[1]][objectness_idx]
            dict_[image_path][names[2]] = dict_[image_path][names_copy[2]][objectness_idx]
            
            if calib_location=="before_nms":
                dict_[image_path][names[3]] = dict_[image_path][names_copy[3]][objectness_idx]
                iou_cross = box_iou(
                    torch.tensor(dict_[image_path][names[3]], device=device),
                    torch.tensor(dict_[image_path]["annot_bbox_xyxy_scaled"], device=device)
                ).cpu().numpy()
            else:
                iou_cross = box_iou(
                    torch.tensor(dict_[image_path][names[0]], device=device),
                    torch.tensor(dict_[image_path]["annot_bbox_xyxy_scaled"], device=device)
                ).cpu().numpy()

            # We want to check if there is an annotated bbox that has the minimum IoU with a predicted bbox.
            if obj_calib is True:
                iou_cross_bool = (iou_cross > iou_thres_obj).astype(int)
                # If there are no annotated bboxs, then all predicted bboxs are wrong.
                if iou_cross_bool.shape[1] == 0:
                    dict_[image_path][names[1]+"_obj_y_true"] = np.zeros(len(objectness_idx)).ravel()
                else:
                    dict_[image_path][names[1]+"_obj_y_true"] = np.max(iou_cross_bool, axis=1).astype(np.int32).ravel()
            
            # For each class, we want to check if there is an annotated bbox that has the minimum IoU with a predicted bbox, then it's accurate for the class.
            if class_calib is True:
                iou_cross_bool = (iou_cross > iou_thres_class).astype(int)
                if iou_cross_bool.shape[1] == 0:
                    dict_[image_path][names[2]+"_class_y_true"] = np.full((num_classes, num_obj_), 0).tolist()
                else:
                    obj_y_true = []
                    true_label = dict_[image_path]["annot_class"]
                    for number in range(num_classes):
                        true_label_bool = (true_label==number)
                        iou_and_true_label = np.max((iou_cross_bool*true_label_bool.reshape(-1, 1).T), axis=1).astype(np.int32)
                        obj_y_true.append(iou_and_true_label)
                    dict_[image_path][names[2]+"_class_y_true"] = obj_y_true
    return names   


def collect_data_obj(dict_, names):
    """
    Collecting all the objectness confidences as a single object to calibrate.

    Parameters
    ----------
    chosen_dict : _type_
        Dictionnary where values to calibrate are saved.
    chosen_dict_obj : _type_
        Dictionnary where the "true" value to use for calibration are saved.
    where_apply : str, optional
        Determines the name of the observations to use when calibrating, by default "pred"

    Returns
    -------
    _type_
        Two arrays with all the uncalibrated y's and "true" y's.
        If plots is true, then, the size of the bboxs and the image id are also saved.
    """
    obj_conf = names[1]
    obj_y_pred, obj_y_true = [], []
    for path in dict_:
        values_ = dict_[path]
        if obj_conf in values_.keys():
            obj_y_true.extend(values_[obj_conf+"_obj_y_true"])
            obj_y_pred.extend(values_[obj_conf].ravel())    
    return obj_y_true, obj_y_pred


def fitting_obj_calibrators(y_true_calib, y_pred_calib, calibrator):
    """
    Fitting the calibrator for each 

    Parameters
    ----------
    y_true_calib : _type_
        The true values to be used as the dependent variable in the calibration.
    y_pred_calib : _type_
        The X values to be used in the calibration.
    calibrator : _type_, optional
        The calibrator to be used for calibration of the values

    Returns
    -------
    _type_
        Fitted calibrator on the calibration values.
    """
    calibrator.fit(y_pred_calib, y_true_calib)
    return calibrator

def predict_obj_conf(dict_, calibrators_fitted, names):
    """
    Objectness predictions using the fitted calibrator.

    Parameters
    ----------
    chosen_dict : _type_
        Dictionnary where to get the predictions to be calibrated.
    chosen_dict_obj : _type_
        Dictionnary where you want to save the calibrated values.
    calibrators_fitted : _type_
        Fitted calibrator.
    where_apply : str, optional
        Determines the name of the observations to use when calibrating, by default "pred"

    Returns
    -------
    _type_
        Calibrated objectness values.
    """
    obj_conf = names[1]
    all_y_calib = []
    for path in dict_:
        values_ = dict_[path]
        if obj_conf in values_.keys():
            obj_score_ = values_[obj_conf]
            preds_ = calibrators_fitted.predict(obj_score_)
            dict_[path][obj_conf+"_calib_obj_score"] = preds_.reshape(-1, 1)
            all_y_calib.extend(list(preds_))
    return all_y_calib


def get_binning_groups(y_score, num_bins, strategy):
    """_summary_

    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.

    Returns
    -------
    _type_
        Returns the upper and lower bound values for the bins and the indices
        of the y_score that belong to each bins.
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, num_bins)
        bins = np.percentile(y_score, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, num_bins)
    elif strategy == "array split":
        bin_groups = np.array_split(y_score, num_bins)
        bins = np.sort(np.array([bin_group.max() for bin_group in bin_groups[:-1]]+[np.inf]))
    else:
        ValueError("We don't have this strategy")
    bin_assignments = np.digitize(y_score, bins, right=True)
    return bins, bin_assignments


def calc_bins(y_score, y_true, num_bins, strategy):
    """
    For each bins, calculate the accuracy, average confidence and size.

    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.

    Returns
    -------
    _type_
        Multiple arrays, the upper and lower bound of each bins, indices of 
        y that belong to each bins, the accuracy, confidecne and size of each bins.
    """
    y_score, y_true = np.array(y_score), np.array(y_true)
    bins, binned = get_binning_groups(y_score, num_bins, strategy)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
        
    for bin in range(num_bins):
        bin_sizes[bin] = len(y_score[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (y_true.reshape(-1, 1)[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (y_score[binned==bin]).sum() / bin_sizes[bin]
    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(y_score, y_true, num_bins, strategy):
    """
    Function to get the different metrics of interest.

    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.

    Returns
    -------
    _type_
        The score of ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
    """
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_score, y_true, num_bins, strategy)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    return ECE, MCE


# Plots
def draw_reliability_graph(y_score, y_true, num_bins, strategy, title, axs=None):
    """
    Plotting the accuracy and confidence per bins and showing the values of ECE and MCE.

    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.
    title : _type_
        Title to give to the graph
    axs : _type_, optional
        If you want to plot multiple graph next to one another, by default None
    """
    ECE, MCE = get_metrics(y_score, y_true, num_bins, strategy)
    bins, _, bin_accs, _, _ = calc_bins(y_score, y_true, num_bins, strategy)

    if axs is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = axs

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Accuracy')

    ## Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    ax.bar(bins, bins, width=1/(bins.shape[0]+1), alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    ax.bar(bins, bin_accs, width=1/(bins.shape[0]+1), alpha=1, edgecolor='black', color='b')
    ax.plot([0,1],[0,1], '--', color='gray', linewidth=2)
    ax.set_title(title)
    
    ab = AnnotationBbox(
        TextArea(
            f"ECE: {np.round(ECE*100, 2)}%\n"
            + f"MCE: {np.round(MCE*100, 2)}%"
        ),
        xy=(0.2, 0.9),
    )
    
    ax.add_artist(ab)
    
    if axs is None:
        plt.show()

def create_pred_for_nms(values_, name_preds, device):
    """
    Re-creating the predictions using the calibrated values.

    Parameters
    ----------
    path : _type_
        The path to the image that you want to create the NMS preciction.
    chosen_dict : _type_
        Dictionnary where you can find the values for the bbox and class.
    chosen_dict_obj : _type_
        Dictionnary where you can find the calibrated values for objectness.
    bbox_name : _type_
        Name of the variable for which you should look for bbox coordinate values. 
    obj_name : _type_
        Name of the variable for which you should look for objectness confidence values. 
    class_name : _type_
        Name of the variable for which you should look for class confidence values. 

    Returns
    ------
    _type_
        The tensor prediction object for this image path.
    """
    bbox_ = torch.tensor(values_[name_preds[0]], device=device)
    obj_ = torch.tensor(values_[name_preds[1]].reshape(-1, 1), device=device)
    class_ = torch.tensor(values_[name_preds[2]], device=device)
    
    pred_ = torch.cat((bbox_, obj_, class_), dim=1)
    pred_ = torch.reshape(pred_, [1, pred_.shape[0], pred_.shape[1]])
    return pred_

def NMS(names, dataloader, dict_, name_preds, num_classes, opt, device):
    """
    Create the tensor and pass the tensor through the NMS

    Parameters
    ----------
    dataloader : _type_
        The dataloader from loading all the pictures.
    chosen_dict : _type_
        Dictionnary where you can find the values for the bbox and class.
    chosen_dict_obj : _type_
        Dictionnary where you can find the calibrated values for objectness.
    before_nms : list, optional
        List with the names of the variables for which you should look for coordinates of the bbox, the objectness and class confidence values, by default ["pred_bbox", "calib_obj_conf", "pred_class_conf"]
    save_after_nms : str, optional
        The name of the saved variables in the dictionnary, by default "after_nms_calib"
    """
    names_copy = names.copy()
    for im, targets, path, shapes in tqdm(dataloader):
        names = names_copy.copy()
        path = path[0]
        if (path in dict_ and dict_[path]["has_detec"]==1):
            if device.type != 'cpu':
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if opt.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

            pred_ = create_pred_for_nms(dict_[path], name_preds, device=device)

            preds_nms_ = non_max_suppression(
                pred_,
                opt.conf_thres_nms,
                opt.iou_thres_nms,
                labels=[],
                multi_label=opt.multi_label_nms,
                agnostic=opt.agnostic_nms,
                max_det=opt.max_det_nms,
                output_confs=True
            )

            # Statistics per image
            for si, pred_nms_ in enumerate(preds_nms_):
                scale_boxes(im[si].shape[1:], pred_nms_[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            names[0] = names[3]+"_scaled"
            get_data_pred(names, preds_nms_[0], dict_, path, num_classes)


def collect_data_class(dict_, num_classes, names):
    """
    Collecting all the class confidences as a single object to calibrate.
    """
    class_conf = names[2]
    obj_y_true = np.zeros((0, num_classes))
    obj_y_pred = np.zeros((0, num_classes))    
    for path in dict_:
        values_ = dict_[path]
        if class_conf in values_.keys():
            obj_y_true = np.vstack([obj_y_true, np.vstack(values_[class_conf+"_class_y_true"]).T])
            obj_y_pred = np.vstack([obj_y_pred, np.vstack(values_[class_conf])])
    return obj_y_true, obj_y_pred


def my_resample(X, y, perc):
    """
    Resampling method before calibration.

    Parameters
    ----------
    X : _type_
        The values that will be used as the training X.
    y : _type_
        The values that will be used as the training X.
    perc : _type_
        Percentage of true values that you want to have in the resampled X and y.

    Returns
    -------
    _type_
        X_res and y_res resampled according to the method defined in the function.
    """
    n = len(y)
    y1_idx = np.where(y==1)[0]
    y0_idx = np.where(y==0)[0]

    if n<=0:
        return X, y
    elif len(y1_idx)/n>0.5:
        return X, y
    else:
        n_perc = int((1-perc)*len(y0_idx))
        y0_idx_chosen = np.random.choice(y0_idx, size=n_perc, replace=False)
        idx = np.append(y0_idx_chosen, y1_idx)
        y_res = y[idx]
        X_res = X[idx]
    
        #sm = imblearn.over_sampling.SMOTE(sampling_strategy=perc)
        #X_res, y_res = sm.fit_resample(X.reshape(-1, 1), y.reshape(-1, 1))
        return X_res, y_res

def fitting_class_calibrators(y_true_calib, y_pred_calib, calibrator, num_classes, perc=0.5):
    """
    Fitting calibrators for the classes.

    Parameters
    ----------
    y_true_calib : _type_
        The true values to be used as the dependent variable in the calibration.
    y_pred_calib : _type_
        The X values to be used in the calibration.
    calibrator : _type_, optional
        The calibrator to be used for calibration of the values
    num_classes : int, optional
        The number of classes that can be predicted in the dataset, by default 4
    perc : float, optional
        Percentage of true values that you want to have in the resampled X and y, by default 0.5

    Returns
    -------
    _type_
        List of 4 estimators, the calibrators for each class.
    """        
    calibrators = []
    for number in range(num_classes):
        cloned_calibrator = clone(calibrator)
        X, y = my_resample(y_pred_calib[:, number], y_true_calib[:, number], perc=perc)
        cloned_calibrator.fit(X, y)
        calibrators.append(cloned_calibrator)
    return calibrators

def predict_class_conf(dict_, calibrators_fitted, num_classes, names):
    """
    Class predictions using the fitted calibrators. 

    Parameters
    ----------
    chosen_dict_obj : _type_
        Dictionnary where to get the predictions to be calibrated.
    chosen_dict_class : _type_
        Dictionnary where you want to save the calibrated values.
    calibrators_fitted : _type_
        Fitted calibrator for each class.
    where_apply : str, optional
        Determines the name of the observations to use when calibrating, by default "after_nms_calib"
    before_nms : _type_, optional
        If you want to calibrate before NMS, then use this dictionnary to show where to find the class confidence values, by default None
    num_classes : int, optional
        The number of classes that can be predicted in the dataset, by default 4

    Returns
    -------
    _type_
        Calibrated class confidence values.
    """
    class_conf = names[2]
    all_y_calib = {}
    for number in range(num_classes):
        all_y_calib[number]= []

    for path in dict_:
        values_ = dict_[path]
        if class_conf in values_.keys():
            pred_values_ = dict_[path][class_conf]
            class_y_calib = []
            for number in range(num_classes):
                preds_ = calibrators_fitted[number].predict(pred_values_[:, number])
                class_y_calib.append(preds_)
                all_y_calib[number].extend(list(preds_))
            dict_[path][class_conf+"_calib_class_score"] = np.vstack(class_y_calib).T
    return np.vstack(list(all_y_calib.values())).T

def get_calibrator(name):
    available_calibrators = {
        "isotonic": IsotonicRegression(out_of_bounds="clip"), # Isotonic Regression
        "platt": _SigmoidCalibration(),
    }
    return available_calibrators[name]

def calibration(
    names,
    calib_location,
    calib_obj,
    calib_class,
    calib_dict,
    test_dict,
    calibrator,
    conf_thres,
    iou_thres_obj,
    iou_thres_class,
    opt,
    num_classes,
    device,
    plots,
    ):
    if (calib_obj is True) or (calib_class is True):
        names_copy = names.copy()
        names = calib_prep(
            names,
            calib_dict,
            num_classes=num_classes,
            device=device,
            conf_thres=conf_thres,
            iou_thres_obj=iou_thres_obj,
            iou_thres_class=iou_thres_class,
            obj_calib=calib_obj,
            class_calib=calib_class,
            calib_location=calib_location,
        )

        _ = calib_prep(
            names_copy,
            test_dict,
            num_classes=num_classes,
            device=device,
            conf_thres=conf_thres,
            iou_thres_obj=iou_thres_obj,
            iou_thres_class=iou_thres_class,
            obj_calib=calib_obj,
            class_calib=calib_class,
            calib_location=calib_location,
        )
        
        names_copy_deep = names.copy()

        # Calibration OBJ
        if calib_obj:
            print("Calibrating objectness score")
            names_copy = names.copy()
            obj_y_true_CALIB, obj_y_pred_CALIB = collect_data_obj(calib_dict, names=names)
            obj_fitted_calibrators = fitting_obj_calibrators(obj_y_true_CALIB, obj_y_pred_CALIB, calibrator)
            obj_y_pred_CALIBRATED = predict_obj_conf(test_dict, obj_fitted_calibrators, names=names)
            names[1] = names[1]+"_calib_obj_score"
            if plots is not None:
                fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(12, 6))
                obj_y_true_TEST, obj_y_pred_TEST = collect_data_obj(test_dict, names=names_copy)
                draw_reliability_graph(obj_y_pred_TEST, obj_y_true_TEST, num_bins=opt.num_bins, strategy="uniform", title="Original objectness", axs=axs1)
                draw_reliability_graph(obj_y_pred_CALIBRATED, obj_y_true_TEST, num_bins=opt.num_bins, strategy="uniform", title="Calibrated objectness", axs=axs2)
                sav_fig_name = os.path.join(plots, "plot_ece_obj_"+calib_location+".png")
                plt.savefig(sav_fig_name)
                plt.close()

        if calib_class:
            print("Calibrating classe scores")
            names_copy = names.copy()
            class_y_true_CALIB, class_y_pred_CALIB = collect_data_class(calib_dict, num_classes=num_classes, names=names)
            class_fitted_calibrators = fitting_class_calibrators(class_y_true_CALIB, class_y_pred_CALIB, calibrator, num_classes=num_classes, perc=0.6)
            class_y_pred_CALIBRATED = predict_class_conf(test_dict, class_fitted_calibrators, num_classes, names=names)
            names[2] = names[2]+"_calib_class_score"
            if plots is not None:
                for i in range(num_classes):
                    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(12, 6))
                    class_y_true_TEST, class_y_pred_TEST = collect_data_class(test_dict, num_classes=num_classes, names=names_copy)
                    draw_reliability_graph(class_y_pred_TEST[:, i], class_y_true_TEST[:, i], num_bins=opt.num_bins, strategy="uniform", title="Original objectness", axs=axs1)
                    draw_reliability_graph(class_y_pred_CALIBRATED[:, i], class_y_true_TEST[:, i], num_bins=opt.num_bins, strategy="uniform", title="Calibrated objectness", axs=axs2)
                    sav_fig_name = os.path.join(plots, "plot_ece_class"+str(i)+"_"+calib_location+".png")
                    plt.savefig(sav_fig_name)
                    plt.close()

        if calib_location=="before_nms":
            if calib_obj:
                _ = predict_obj_conf(calib_dict, obj_fitted_calibrators, names=names_copy_deep)
            if calib_class:
                _ = predict_class_conf(calib_dict, class_fitted_calibrators, num_classes, names=names_copy_deep)
        return names
    else:
        for image_path in calib_dict:
            len_obj_ = len(np.where(calib_dict[image_path][names[1]] > conf_thres)[0])
            if len_obj_>0:
                calib_dict[image_path]["has_detec"] = 1
            else:
                calib_dict[image_path]["has_detec"] = 0
        for image_path in test_dict:
            len_obj_ = len(np.where(test_dict[image_path][names[1]] > conf_thres)[0])
            if len_obj_>0:
                test_dict[image_path]["has_detec"] = 1
            else:
                test_dict[image_path]["has_detec"] = 0
        return names
            

def get_annotations_from_dict(dict_, device):
    labels = []
    for path in dict_:
        values_ = dict_[path]
        if values_["has_detec"]==1:
            label_ = torch.cat(
                (
                    torch.tensor(values_["annot_class"], device=device),
                    torch.tensor(values_["annot_bbox_xyxy_scaled"], device=device),
                ),
                dim=1
            )
            labels.append(label_)
    labels_numpy = np.vstack([x.clone().cpu().numpy() for x in labels])
    return labels, labels_numpy


def get_preds_from_dict(dict_, names, device):
    detections = []
    for path in dict_:
        values_ = dict_[path]
        if values_["has_detec"]==1:
            detection_ = torch.cat(
                (
                    torch.tensor(values_[names[0]], device=device),
                    torch.tensor(values_[names[1]], device=device),
                    torch.tensor(values_[names[2]], device=device),
                ),
                dim=1
            )
            detections.append(detection_)
    detections_numpy = np.vstack([x.clone().cpu().numpy() for x in detections])
    return detections, detections_numpy



def calc_mAP(target_dir, names, device, title, plots=False):
    test_dict = {}
    file_test = os.path.join(target_dir, "test_dict.pickle")
    if os.path.getsize(file_test) > 0:      
        with open(file_test, "rb") as f:
            unpickler = pickle.Unpickler(f)
            test_dict = unpickler.load()

    # names = ["after_nms"+"_bbox_xyxy_scaled", "after_nms"+"_obj_score", "after_nms"+"_class_score"]
    test_values, _ = get_preds_from_dict(test_dict, names, device)
    annotation_values, _ = get_annotations_from_dict(test_dict, device)

    file_var = os.path.join(target_dir, "var.yaml")
    with open(file_var) as f:
        var = Namespace(**yaml.safe_load(f))

    names = var.names
    nc = var.nc
    confusion_matrix = ConfusionMatrix(nc=nc)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = []
    seen = 0

    for si, test_ in enumerate(test_values):
        annot_ = annotation_values[si]
        labels = annot_[annot_[:, 0] == si, 1:]
        nl, npr = labels.shape[0], test_.shape[0]  # number of labels, predictions
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        seen += 1

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                if plots:
                    confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            continue

        argmax_ = test_[:, 5:].clone().argmax(axis=1).reshape(-1, 1)
        testn = torch.cat((test_[:, :5], argmax_), 1)
        correct = process_batch(testn, annot_, iouv)
        stats.append((correct, testn[:, 4], testn[:, 5], annot_[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

    with open(f'{target_dir}/{title}.txt', "a") as file_text:
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        file_text.write(s)
        LOGGER.info(s)
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        
        # Print results
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        s = pf % ('all', seen, nt.sum(), mp, mr, map50, map)
        file_text.write(s)
        LOGGER.info(s)
        if nt.sum() == 0:
            LOGGER.warning(f'WARNING ⚠️ no labels found in calibration set, can not compute metrics without labels')

        # Print results per class
        if nc < 50 and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                s = pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])
                file_text.write(s)
                LOGGER.info(s)