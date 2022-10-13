import numpy as np
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

from models.common import DetectMultiBackend
from utils.general import (check_dataset, xywh2xyxy, Profile, check_img_size, colorstr, scale_boxes, non_max_suppression)
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader

import torch

def get_data_pred(preds_, chosen_dict, image_path, name, num_classes):
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
    chosen_dict[image_path][name+'_bbox_xyxy_scaled'] = preds_[:, :4].clone().cpu().numpy()
    chosen_dict[image_path][name+'_obj_score'] = preds_[:, -num_classes-1:-num_classes].clone().cpu().numpy()
    chosen_dict[image_path][name+'_class_score'] = preds_[:, -num_classes:].clone().cpu().numpy()


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
        data_dict[paths[0]]['pred_bbox_xywh'] = preds[0][:, :4].clone().cpu().numpy()
        get_data_pred(preds_[0], data_dict, paths, "before_nms", num_classes=nc)
        

        for si, pred_nms_ in enumerate(preds_nms):
            scale_boxes(im[si].shape[1:], pred_nms_[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Normal predictions after NMS
        get_data_pred(preds_nms[0], data_dict, paths, "after_nms", num_classes=nc)

        # True images
        data_dict[paths[0]]['true_image_shape'] = (height, width)
        data_dict[paths[0]]['true_bbox_xyxy_scaled'] = tbox.clone().cpu().numpy()
        data_dict[paths[0]]['true_bbox'] = targets[:, 2:].clone().cpu().numpy()
        data_dict[paths[0]]['true_class'] = targets[:, 1:2].clone().cpu().numpy()
    return data_dict


# Create dictionnary with the new labels, the y_true
def calib_prep(dict_, where_apply_calib, num_classes, device, conf_thres=0.02, iou_thres_obj=0, iou_thres_class=0, obj_calib=True, class_calib=False):
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
    bbox_pred = where_apply_calib+"_bbox_xyxy_scaled"
    obj_conf = where_apply_calib+"_obj_score"
    class_conf = where_apply_calib+"_class_score"        
                
    for i, image_path in enumerate(tqdm(dict_)):
        objectness_idx = np.where(dict_[image_path][obj_conf] > conf_thres)[0]
        num_obj_ = len(objectness_idx)

        # If no bboxs have the matching criterion, then forget this image
        if num_obj_==0:
            pass
        else:
            dict_[image_path]["idx"] = objectness_idx
            dict_[image_path]["pred_bbox_xywh"+"_idx"] = dict_[image_path]["pred_bbox_xywh"][objectness_idx]
            dict_[image_path][obj_conf+"_idx"] = dict_[image_path][obj_conf][objectness_idx]
            dict_[image_path][class_conf+"_idx"] = dict_[image_path][class_conf][objectness_idx]
            
            iou_cross = box_iou(
                torch.tensor(dict_[image_path][bbox_pred][objectness_idx], device=device),
                torch.tensor(dict_[image_path]["true_bbox_xyxy_scaled"], device=device)
            ).cpu().numpy()

            # We want to check if there is an annotated bbox that has the minimum IoU with a predicted bbox.
            if obj_calib is True:
                iou_cross_bool = (iou_cross > iou_thres_obj).astype(int)
                # If there are no annotated bboxs, then all predicted bboxs are wrong.
                if iou_cross_bool.shape[1] == 0:
                    dict_[image_path][where_apply_calib+"_obj_y_true"] = np.zeros(len(objectness_idx)).ravel()
                else:
                    dict_[image_path][where_apply_calib+"_obj_y_true"] = np.max(iou_cross_bool, axis=1).astype(np.int32).ravel()
            
            # For each class, we want to check if there is an annotated bbox that has the minimum IoU with a predicted bbox, then it's accurate for the class.
            if class_calib is True:
                iou_cross_bool = (iou_cross > iou_thres_class).astype(int)
                if iou_cross_bool.shape[1] == 0:
                    dict_[image_path][where_apply_calib+"_class_y_true"] = np.full((num_classes, num_obj_), 0).tolist()
                else:
                    obj_y_true = []
                    true_label = dict_[image_path]["true_class"]
                    for number in range(num_classes):
                        true_label_bool = (true_label==number)
                        iou_and_true_label = np.max((iou_cross_bool*true_label_bool.reshape(-1, 1).T), axis=1).astype(np.int32)
                        obj_y_true.append(iou_and_true_label)
                    dict_[image_path][where_apply_calib+"_class_y_true"] = obj_y_true


def collect_data_obj(dict_, where_apply_calib):
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
    obj_conf = where_apply_calib+"_obj_score"+"_idx"
    obj_y_pred, obj_y_true = [], []
    for path in dict_:
        values_ = dict_[path]
        if obj_conf in values_.keys():
            obj_y_true.extend(values_[where_apply_calib+"_obj_y_true"])
            obj_y_pred.extend(values_[obj_conf].ravel())    
    return obj_y_pred, obj_y_true


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

def predict_obj_conf(dict_, calibrators_fitted, where_apply_calib):
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
    obj_conf = where_apply_calib+"_obj_score"+"_idx"
    all_y_calib = []
    for path in dict_:
        values_ = dict_[path]
        if obj_conf in values_.keys():
            obj_score_ = values_[obj_conf]
            preds_ = calibrators_fitted.predict(obj_score_)
            dict_[path][where_apply_calib+"_calib_obj_score"] = preds_.reshape(-1, 1)
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
def draw_reliability_graph(y_score, y_true, num_bins, strategy, title, axs=None, sav=None):
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
    
    if sav is not None:
        plt.savefig(sav+".png")
    
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

def NMS(dataloader, dict_, name_preds, save_after_nms, num_classes, opt, device):
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
    for im, targets, path, shapes in tqdm(dataloader):
        path = path[0]
        if path in dict_:
            values_ = dict_[path]
            if "idx" in values_.keys():
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

                get_data_pred(preds_nms_[0], dict_, path, save_after_nms, num_classes)






def collect_data_class(dict_, num_classes, where_apply_calib):
    """
    Collecting all the class confidences as a single object to calibrate.
    """
    class_conf = where_apply_calib+"_class_score"+"_idx"
    obj_y_true = np.zeros((0, num_classes))
    obj_y_pred = np.zeros((0, num_classes))
    
    for path in dict_:
        values_ = dict_[path]
        if class_conf in values_.keys():
            obj_y_true = np.vstack([obj_y_true, np.vstack(values_[where_apply_calib+"_class_y_true"]).T])
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

    if len(y1_idx)/n>0.5:
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

def predict_class_conf(dict_, calibrators_fitted, num_classes, where_apply_calib):
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
    class_conf = where_apply_calib+"_class_score"+"_idx"
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
            dict_[path][where_apply_calib+"_calib_class_score"] = np.vstack(class_y_calib).T

    return np.vstack(list(all_y_calib.values())).T

def get_calibrator(name):
    available_calibrators = {
        "isotonic": IsotonicRegression(out_of_bounds="clip"), # Isotonic Regression
        "platt": _SigmoidCalibration(),
    }
    return available_calibrators[name]

def calibration(
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
    opt,
    num_classes,
    device
    ):
    if (calib_obj is True) or (calib_class is True):
        calib_prep(
            calib_dict,
            where_apply_calib=calib_location,
            num_classes=num_classes,
            device=device,
            conf_thres=conf_thres,
            iou_thres_obj=iou_thres_obj,
            iou_thres_class=iou_thres_class,
            obj_calib=calib_obj,
            class_calib=calib_class
        )

        calib_prep(
            test_dict,
            where_apply_calib=calib_location,
            num_classes=num_classes,
            device=device,
            conf_thres=conf_thres,
            iou_thres_obj=iou_thres_obj,
            iou_thres_class=iou_thres_class,
            obj_calib=calib_obj,
            class_calib=calib_class
        )

        # Calibration OBJ
        if calib_obj:
            print("calib obj")
            obj_y_pred_CALIB, obj_y_true_CALIB = collect_data_obj(calib_dict, where_apply_calib=calib_location)
            fitted_calibrator = fitting_obj_calibrators(obj_y_true_CALIB, obj_y_pred_CALIB, calibrator)
            _ = predict_obj_conf(test_dict, fitted_calibrator, where_apply_calib=calib_location)

        if calib_class:
            print("calib class")
            list_y_true_calib, list_y_pred_calib = collect_data_class(calib_dict, num_classes=num_classes, where_apply_calib=calib_location)
            calibrators_fitted = fitting_class_calibrators(list_y_true_calib, list_y_pred_calib, calibrator, num_classes=num_classes, perc=0.6)
            _ = predict_class_conf(test_dict, calibrators_fitted, num_classes, calib_location)

        if calib_location=="before_nms":
            print("running NMS")
            if calib_obj:
                _ = predict_obj_conf(calib_dict, fitted_calibrator, where_apply_calib=calib_location)
                name_pred_obj = calib_location+"_calib_obj_score"
            else:
                name_pred_obj = calib_location+"_obj_score_idx"

            if calib_class:
                _ = predict_class_conf(calib_dict, calibrators_fitted, num_classes, calib_location)
                name_pred_class = calib_location+"_calib_class_score"
            else:
                name_pred_class = calib_location+"_class_score_idx"

            name_preds = ["pred_bbox_xywh_idx", name_pred_obj, name_pred_class]
            print("nms calib")
            NMS(dataloader, calib_dict, name_preds=name_preds, save_after_nms="after_nms_calib", num_classes=num_classes, opt=opt, device=device)

            print("nms test")
            NMS(dataloader, test_dict, name_preds=name_preds, save_after_nms="after_nms_calib", num_classes=num_classes, opt=opt, device=device)
            
