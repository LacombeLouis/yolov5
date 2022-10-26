import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox


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