"""Spectroscopy plots and style plots"""
from inspect import isdatadescriptor
import sys
import os.path
import pathlib
import time


import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import style
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import ipywidgets as widgets
from lmfit import Parameters, Model
from lmfit.model import load_model
from lmfit.models import LorentzianModel, ThermalDistributionModel
from tables import file


from .h5utils import criteria_name, is_dataset, is_group, h5Utils
from .analysis import find_near, array_region, fit_baseline, nm_to_ev


###############################################################################
#STYLES
###############################################################################
colors = list(mplc.BASE_COLORS.values())

linestyle_tuple = [
     #('loosely dotted',        (0, (1, 10))),
     #('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     #('long dash with offset', (5, (10, 3))),
     #('loosely dashed',        (0, (5, 10))),
     #('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     #('loosely dashdotted',    (0, (3, 10, 1, 10))),
     #('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     #('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     #('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ]

linestyles = [ls for (name, ls )in linestyle_tuple]
markers = [
        "v",
        "^",
        ">",
        "<",
        "1",
        "2",
        "3",
        "4",
        ] 

#Spectrscopy plots
###############################################################################
def spectra(dataset, style='', save= '' , attributes=[], baseline='',  **kwargs):
    """
    Plot a spectra from an HDF5 dataset.

    Parameters:
    -----------
    dataset : h5py.Dataset or numpy.ndarray
        The dataset containing spectral data. If it's an h5py dataset, it is expected to have two columns:
        the first column for energy values (in eV), and the second column for intensity (counts).

    style : str, optional
        The style to be applied to the plot. If provided, it should be a valid Matplotlib style name.
        If not specified, the 'default' style will be used.

    save : str, optional
        The file path to save the generated plot as an image. If not provided, the plot will not be saved.

    attributes : list, optional
        A list of attribute names to include in the plot's label. If the dataset is an h5py dataset,
        these attributes will be retrieved and added to the label.

    baseline : str, optional
        The name of an attribute that can be used as a baseline for the spectral data. If specified, this
        attribute will be used as the intensity data, and the energy data will be taken from the first column
        of the dataset.

    **kwargs : additional keyword arguments
        Additional keyword arguments to be passed to the Matplotlib `ax.plot` function for customizing the plot.

    Returns:
    --------
    None

    Examples:
    ---------
    # Plot a spectrum from an HDF5 dataset with custom styling and save the plot.
    spectra(hdf5_dataset, style='seaborn-darkgrid', save='spectrum.png', attributes=['sample_name'])
    """
    label=''
    
    if is_dataset(dataset):
        data = np.array(dataset)
        title = dataset.name
        for attribute in attributes:
            label += f'{attribute}:{dataset.attrs.get(attribute)}\n'
        if baseline:
            data = dataset.attrs.get(baseline) 
    else:
        data=np.array(dataset)

    if style:
        try:plt.style.use(style)
        except NameError: plt.style.use('default')

    fig, ax = plt.subplots(1, 1, constrained_layout= True)
    [l.remove() for l in ax.lines]
    ax.plot(data[:,0],data[:,1], label=label, **kwargs)
    ax.set_xlabel('Energy(eV)')
    ax.set_ylabel('Intensity(counts)')
    ax.minorticks_on()
    if label:ax.legend()
    if save: plt.savefig(save, dpi=dpi, transparent= True)

    return None


def interactive_spectra(file_name_or_object, keys, mode='r',
                style='', attributes=[], baseline='', **kwargs):
    """
    Create an interactive spectral plot for HDF5 files.

    This function is designed to work specifically with HDF5 files. It allows you to visualize spectra
    stored in an HDF5 file interactively by selecting different datasets using a slider.

    Parameters:
    -----------
    file_name_or_object : str or h5py.File or h5py.Group
        The name of the HDF5 file or an open h5py File/Group object to read data from. 

    keys : tuple
        A tuple containing keys corresponding to datasets within the HDF5 file. These keys are used
        to select different spectra for interactive plotting.

    mode : str, optional
        The file access mode for reading the HDF5 file. Default is 'r' (read-only).

    style : str, optional
        The Matplotlib style to apply to the interactive plot. If not specified, the default style is used.

    attributes : list, optional
        A list of attribute names to include in the plot's label when displaying spectra information.

    baseline : str, optional
        The name of an attribute in the HDF5 dataset that can be used as a baseline for spectral data.

    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the underlying `spectra` function when plotting.

    Returns:
    --------
    None

    Note:
    -----
    Remember to close the HDF5 file using the `close` method of the h5py File object after you have finished
    using it.

    Example:
    --------
    # Create an interactive spectral plot for an HDF5 file with multiple spectra in jupyter notebook)
    #With file name
    sample = interactive_spectra('data.h5', keys=('spectrum_1', 'spectrum_2', 'spectrum_3'), style='ggplot',
                        attributes=['sample_name', 'temperature'], baseline='background_intensity')
    #use interative figure and close
    sample.close

    #With object file
    keys = ('spectrum_1', 'spectrum_2', 'spectrum_3')
    sample = h5py.File(file_name, 'r')
    interactive_spectra(sample, keys, attributes=['sample_name', 'temperature'])
    #use interactive figure and then close 
    sample.close()
    """

    if isinstance(file_name_or_object, str):
        sample = h5py.File(file_name_or_object, mode=mode)
    elif isinstance(file_name_or_object, h5py.File):
        sample = file_name_or_object

    #with sample.access_h5() as h5_obj:

    @widgets.interact(key=(0, len(keys) - 1 , 1))
    def plot(key=0):
        """Remove old lines from plot and plot net one"""
        dataset = sample.get(keys[key])
        spectra(dataset, style=style, attributes=attributes, baseline=baseline)
        return None
        
        
    return sample








#Other
###############################################################################
def ImageZooms(image, Px, Py, dx, dy, labels,
        figsize=(6.3,6.3/2), mplstyle=['default'], cmap = 'gist_gray',
        xlim=[], ylim = [], save=''):
    """
    ImageZooms
    Display an image along with corresponding zoomed-in images indicated by positions (Px, Py) and dimensions (dx, dy).

    Parameters:
    -----------
    image : str or numpy.ndarray
        The image to be displayed. It can be either the path to a JPG image or a numpy.ndarray containing the image data.

    Px : list
        List of x positions of the zoomed images, e.g., [x1, x2, ..., xn].

    Py : list
        List of y positions of the zoomed images, e.g., [y1, y2, ..., yn].

    dx : list
        List of x dimensions (size) of the zoomed images, e.g., [dx1, dx2, ..., dxn].

    dy : list
        List of y dimensions (size) of the zoomed images, e.g., [dy1, dy2, ..., dyn].

    labels : list
        List of labels for the zoomed images, e.g., ['zoom1', 'zoom2', ..., 'zoomn'].

    figsize : tuple, optional
        The size of the figure for the main image and zoomed images. Default is (6.3, 6.3/2).

    mplstyle : str or list, optional
        The Matplotlib style or a list of styles to apply to the plot. If not specified, the default style is used.

    cmap : str, optional
        The colormap to use for displaying the images. Default is 'gist_gray'.

    xlim : list, optional
        The x-axis limits for the main image and zoomed images. Default is to use the full extent of the image.

    ylim : list, optional
        The y-axis limits for the main image and zoomed images. Default is to use the full extent of the image.

    save : str, optional
        The file path to save the generated plot as an image. If not provided, the plot will not be saved.

    Returns:
    --------
    None

    Notes:
    ------
    The parameters Px, Py, dx, dy, and labels must have the same length, indicating the same number of zoomed images.

    Credits:
    --------
    - Title: Scientific Visualization - Python & Matplotlib
    - Author: Nicolas P. Rougier
    - License: BSD

    Example:
    --------
    # Display an image and its zoomed-in regions with labels
    ImageZooms('image.jpg', Px=[100, 200, 300], Py=[100, 150, 200], dx=[50, 60, 70], dy=[50, 60, 70], labels=['Zoom 1', 'Zoom 2', 'Zoom 3'], save='zoomed_image.png')
    """


    if mplstyle: plt.style.use(mplstyle)
    else: plt.style.use(['default'])

    if isinstance(image, type('')):
        image = mpimg.imread(image)
    elif isinstance(image, np.ndarray):
        image = image
    else:
        return None

    fig = plt.figure(figsize=figsize)
    
    shape = np.shape(image) 
    if len(dx)==len(Px) and len(dy)==len(Py) and len(Px)==len(Py):
        n = len(Px)
    else:
        lens = np.array([len(dx), len(dy), len(Px), len(Py)])
        n = int(np.min(lens))

    gs = GridSpec(n, n + 1)

    if xlim: 
        try:
            xa, xb= xlim
            if xa>xb: xa,xb = xlim[::-1]
        except ValueError:
            print('xlim no valid')
            return None 
    else:
        xa, xb = [0, shape[1]-1]
    if ylim: 
        try:
            ya, yb= ylim
            if ya>yb: ya,yb = ylim[::-1]
        except ValueError:
            print('ylim no valid')
            return None 
    else:
        ya, yb = [0, shape[0]-1]

    
    ax = plt.subplot(gs[:n, :n], xlim=[xa, xb],
            xticks=[], ylim=[ya, yb], yticks=[], aspect=1)
    ax.imshow(image, cmap=cmap)

    for i, (x, y) in enumerate(zip(Px, Py)):
        sax = plt.subplot(
            gs[i, n],
            xlim=[x - dx[i], x + dx[i]],
            xticks=[],
            ylim=[y - dy[i], y + dy[i]],
            yticks=[],
            aspect=1,
        )
        sax.imshow(image , cmap=cmap)
    
        sax.text(
            1.1,
            0.5,
            labels[i],
            rotation=90,
            size=8,
            ha="left",
            va="center",
            transform=sax.transAxes,
        )
    
        rect = Rectangle(
            (x - dx[i], y - dy[i]),
            2 * dx[i],
            2 * dy[i],
            edgecolor="black",
            facecolor="None",
            linestyle="--",
            linewidth=0.75,
        )
        ax.add_patch(rect)
    
        con = ConnectionPatch(
            xyA=(x, y),
            coordsA=ax.transData,
            xyB=(0, 0.5),
            coordsB=sax.transAxes,
            linestyle="--",
            linewidth=0.75,
            patchA=rect,
            arrowstyle="->",
        )
        fig.add_artist(con)
    
    
    plt.tight_layout()
    if save: plt.savefig(save)
    return None







