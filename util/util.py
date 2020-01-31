# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Util.py

import json
import os
import scipy.io
import numpy as np
import copy
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.pyplot import cm
from random import seed
from skimage.measure import compare_ssim as ssim
from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D, Conv1D


""" Saving/loading/checking files from disk """

def load_config(config_filepath):
    """ Load a session configuration from a JSON-formatted file.

    Args:
    config_filepath: string
    Returns: dict

    """

    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
    else:
        with config_file:
            return json.load(config_file)

def save_config(config_filepath, config):
    """ Save a session configuration to a JSON-formatted file.

    Args:
    config_filepath: string
    config: dict

    """


    try:
        config_file = open(config_filepath, 'w')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
    else:
        json.dump(config, config_file, indent=4, sort_keys=True)


def dir_contains_files(path):
    """ Check if a directory contains files.

    Args:
    path: string
    Returns: boolean

    """

    for f in os.listdir(path):
        if not f.startswith('.'):
            return True
    return False

def load_soundfield(filepath, freq):
    """ Load a simulated sound field saved in a mat file.

    Args:
    filepath: string
    freq: list

    Returns: np.ndarray

    """

    frequencies = np.asarray(freq)
    mat = scipy.io.loadmat(filepath)
    f_response = mat['AbsFrequencyResponse']
    f_response = np.transpose(f_response, (1, 0, 2))
    soundfield = f_response[:, :, frequencies]
    return soundfield

def load_RoomB_soundfield(filepath, source_position):
    """ Load the measured sound field saved in a mat file.

    Args:
    filepath: string
    freq: int

    Returns: np.ndarray

    """
    mat = scipy.io.loadmat(filepath)
    f_response = mat['AbsFrequencyResponse']
    f_response = f_response[..., source_position]
    soundfield = np.transpose(f_response, (1, 0, 2))
    return soundfield

def get_frequencies():
    """Loads the frequency numbers found at 'util/frequencies.txt'.

    Returns: list

    """

    freqs_path = 'util/frequencies.txt'
    with open(freqs_path) as f:
        freqs = [[int(freq) for freq in line.strip().split(' ')] for line in f.readlines()][0]

    return freqs

""" Data processing functions """

def preprocessing(factor, sf, mask):
    """ Perfom all preprocessing steps.

        Args:
        factor: int
        sf: np.ndarray
        mask: np.ndarray

        Returns: np.ndarray, np.ndarray

        """

    # Downsampling
    downsampled_sf = downsampling(factor, sf)

    # Masking
    masked_sf = apply_mask(downsampled_sf, mask)

    # Scaling masked sound field
    scaled_sf = scale(masked_sf)

    # Upsampling scaled sound field and mask
    irregular_sf, mask = upsampling(factor, scaled_sf, mask)

    return irregular_sf, mask


def downsampling(dw_factor, input_sfs):
    """ Downsamples sound fields given a downsampling factor.

        Args:
        dw_factor: int
        input_sfs: np.ndarray

        Returns: np.ndarray

        """
    return input_sfs[:, 0:input_sfs.shape[1]:dw_factor, 0:input_sfs.shape[2]:dw_factor, :]


def apply_mask(input_sfs, masks):
    """ Apply masks to sound fields.

        Args:
        input_sfs: np.ndarray
        masks: np.ndarray

        Returns: np.ndarray

        """

    masked_sfs = []
    for sf, mk in zip(input_sfs, masks):
        aux_sf = copy.deepcopy(sf)
        aux_sf[mk==0] = 0
        for i in range(sf.shape[2]):
            aux_max = aux_sf[:, :, i].max()
            sf[:, :, i][mk[:, :, i]==0] = aux_max
        masked_sfs.append(sf)
    return np.asarray(masked_sfs)

def scale(input_sfs):
    """ Scale data in range 0-1.

        Args:
        input_sfs: np.ndarray

        Returns: np.ndarray

        """

    scaled_sf = []
    for sf in input_sfs:
        for i in range(sf.shape[2]):
            aux_max = sf[:, :, i].max()
            aux_min = sf[:, :, i].min()
            if aux_max == aux_min:
                sf[:, :, i] = 1
            else:
                sf[:, :, i] = (sf[:, :, i]-aux_min)/(aux_max-aux_min)
        scaled_sf.append(sf)
    return np.asarray(scaled_sf)

def upsampling(up_factor, input_sfs, masks):
    """ Upsamples sound fields and masks given a upsampling factor.

        Args:
        up_factor: int
        input_sfs: np.ndarray
        masks: np.ndarray

        Returns: np.ndarray, np.ndarray

        """

    batch_sf_up = []
    batch_mask_up = []

    for sf, mask in zip(input_sfs, masks): #for each sample in the batch size
        sf_up = []
        mask_up = []
        sf = np.swapaxes(sf, 2, 0)
        mask = np.swapaxes(mask, 2, 0)
        for sf_slice in sf:
            positions = np.repeat(range(1, sf_slice.shape[1]), up_factor-1) #positions in sf slice to put 1
            sf_slice_up = np.insert(sf_slice, obj=positions,values=np.ones(len(positions)), axis=1)
            sf_slice_up = np.transpose(np.insert(np.transpose(sf_slice_up),obj=positions,values=np.ones(len(positions)), axis=1))
            sf_slice_up = np.pad(sf_slice_up, (0,up_factor-1),  mode='constant', constant_values=1)
            sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=0)
            sf_slice_up = np.roll(sf_slice_up, (up_factor-1)//2, axis=1)
            sf_up.append(sf_slice_up)

        mask_slice = mask[0, :, :]
        positions = np.repeat(range(1, mask_slice.shape[1]), up_factor-1) #positions in mask slice to put 0
        mask_slice_up = np.insert(mask_slice, obj=positions,values=np.zeros(len(positions)), axis=1)
        mask_slice_up = np.transpose(np.insert(np.transpose(mask_slice_up),obj=positions,values=np.zeros(len(positions)), axis=1))
        mask_slice_up = np.pad(mask_slice_up, (0,up_factor-1),  mode='constant')
        mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=0)
        mask_slice_up = np.roll(mask_slice_up, (up_factor-1)//2, axis=1)
        mask_slice_up = mask_slice_up[np.newaxis, :]
        mask_up = np.repeat(mask_slice_up, mask.shape[0], axis=0)


        batch_sf_up.append(sf_up)
        batch_mask_up.append(mask_up)

    batch_sf_up = np.asarray(batch_sf_up)
    batch_sf_up = np.swapaxes(batch_sf_up, 3, 1)

    batch_mask_up = np.asarray(batch_mask_up)
    batch_mask_up = np.swapaxes(batch_mask_up, 3, 1)

    return batch_sf_up, batch_mask_up

def postprocessing(pred_sf, measured_sf, freq_num, pattern, factor):
    """ Perfoms all postprocessing steps.

        Args:
        pred_sf: np.ndarray
        measured_sf: np.ndarray
        freq_num: int
        pattern: np.ndarray
        factor: int

        Returns: np.ndarray

        """

    # Use linear regression to compute the rescaling parameters

    measured_sf_slice = copy.deepcopy(measured_sf[0, :, :, freq_num])

    # Downsampling pred_sf to compute regression using same positions.
    pred_sf_dw = downsampling(factor, pred_sf)

    x = np.asarray(pred_sf_dw[0, :, :, freq_num].flatten()[pattern])
    y = np.asarray(measured_sf_slice.flatten()[pattern])

    A = np.vstack([x, np.ones(len(x))]).T

    # compute regression coefficients
    m, c = np.linalg.lstsq(A, y, rcond=-1)[0]

    # rescale values
    reconstructed_sf_slice = pred_sf[0, :, :, freq_num]*m + c

    return reconstructed_sf_slice


""" Evaluation tools and metrics """

def write_results(filepath, results_dict):
    """ Write evaluation results to a .csv file

        Args:
        filepath: string
        results_dict: dict

        Returns: np.ndarray

        """

    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
        writer.writeheader()
        writer.writerows(convert_dict(results_dict, len(results_dict['name'])))

def convert_dict(mydict, numentries):
    """ Convert dict of lists to list of dicts.

        Args:
        mydict: dict
        numentries: int

        Returns: list

        """

    data = []
    for i in range(numentries):
        row = {}
        for k, l in mydict.items():
            row[k] = l[i]
        data.append(row)
    return data

def compute_NMSE(ref, pred):
    """ Compute Normalised Mean Square Error.

        Args:
        ref: np.ndarray
        pred: np.ndarray

        Returns: np.float64

        """

    mse = np.mean((ref.flatten() - pred.flatten()) ** 2)
    return mse / np.mean(ref.flatten()**2)

def compute_SSIM(ref, pred, data_range):
    """ Compute Structural Similarity.

        Args:
        ref: np.ndarray
        pred: np.ndarray
        data_range: np.float64

        Returns: np.float64

        """
    return ssim(ref, pred, data_range=data_range)

def compute_average_pressure(soundfield_slice):
    """ Compute Average Pressure.

        Args:
        soundifeld_slice: np.ndarray

        Returns: np.float64

        """
    p_square = np.mean(np.square(np.absolute(soundfield_slice)))
    return 10.0*np.log10(p_square)

""" Analysis and plotting tools for real sound fields """
def analyze_and_plot_real_results(results_filepath, config):
    """ Read real results .csv files, analyze data, and plot.

        Args:
        results_filepath: string
        config: dict

        """

    plot_path, _ = os.path.split(results_filepath)

    df_all = pd.read_csv(results_filepath)

    freqs = get_frequencies()

    for source in [0, 1]:
        plt.figure(3, figsize = (25, 12.5))
        GLOBAL_NMSE = plt.axes()
        plt.figure(11, figsize = (25, 12.5))
        GLOBAL_SSIM = plt.axes()

        for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
            df1 = df_all[['freq','NMSE', 'SSIM', 'num_mics','num_file', 'num_comb']]
            mic_results = df1.loc[df1['num_file'] == source]
            mic_results = mic_results.loc[mic_results['num_mics'] == num_mics]
            res = mic_results[['NMSE', 'SSIM', 'num_comb']]
            nmse = []
            ssim = []
            for num_comb in range(config['evaluation']['num_comb']):
                #Add data for each combination
                defin = res.loc[res['num_comb'] == num_comb]
                nmse.append(defin['NMSE'].values)
                ssim.append(defin['SSIM'].values)

            nmse = np.asarray(nmse)
            ssim = np.asarray(ssim)

            #plot nmse mic results given all combinations
            label = str(num_mics)
            m, lb, ub = mean_confidence_interval(nmse)
            GLOBAL_NMSE = plot_mean_and_CI(GLOBAL_NMSE, m, lb, ub, label, freqs)

            #plot ssmi mic results given all combinations
            label = str(num_mics)
            m, lb, ub = mean_confidence_interval(ssim)
            GLOBAL_SSIM = plot_mean_and_CI(GLOBAL_SSIM, m, lb, ub, label, freqs)

        pretty_plot(GLOBAL_NMSE, 'NMSE', plot_path, source)
        pretty_plot(GLOBAL_SSIM, 'SSIM', plot_path, source)


def analyze_and_plot_simulated_results(evaluation_path, session_dir, config):
    """ Read simulated results .csv files, analyze data, and plot.

        Args:
        evaluation_path: string
        session_dir: string
        config: dict

        """
    filenames = [filename for filename in os.listdir(evaluation_path) if filename.endswith('.csv')]

    freqs = get_frequencies()

    results = {}

    for metric in ['NMSE', 'SSIM']:
        results[metric] = {}

    for metric in ['NMSE', 'SSIM']:
        for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
            results[metric][num_mics] = []

    for num_file, filename in enumerate(filenames):
        df_all = pd.read_csv(os.path.join(evaluation_path, filename))

        plt.figure(3, figsize = (25, 12.5))
        GLOBAL_NMSE = plt.axes()
        plt.figure(11, figsize = (25, 12.5))
        GLOBAL_SSIM = plt.axes()

        file_results = df_all[['NMSE', 'SSIM', 'num_mics', 'num_comb']]

        for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
            res = file_results.loc[file_results['num_mics'] == num_mics]
            for num_comb in range(config['evaluation']['num_comb']):
                #Add data for each combination
                defin = res.loc[res['num_comb'] == num_comb]
                results['NMSE'][num_mics].append(defin['NMSE'].values)
                results['SSIM'][num_mics].append(defin['SSIM'].values)

    for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
        nmse = np.asarray(results['NMSE'][num_mics])
        ssim = np.asarray(results['SSIM'][num_mics])

        #plot nmse mic results given all combinations
        label = str(num_mics)
        m, lb, ub = mean_confidence_interval(nmse)
        GLOBAL_NMSE = plot_mean_and_CI(GLOBAL_NMSE, m, lb, ub, label, freqs)

        #plot ssmi mic results given all combinations
        label = str(num_mics)
        m, lb, ub = mean_confidence_interval(ssim)
        GLOBAL_SSIM = plot_mean_and_CI(GLOBAL_SSIM, m, lb, ub, label, freqs)

    pretty_plot(GLOBAL_NMSE, 'NMSE', evaluation_path)
    pretty_plot(GLOBAL_SSIM, 'SSIM', evaluation_path)


def mean_confidence_interval(data, confidence=0.95):
    """ Compute data mean and confidence boundaries.

        Args:
        data: np.ndarray
        confidence: float

        Returns: np.ndarray, np.ndarray, np.ndarray

        """
    data = 1.0 * np.array(data)
    n = len(data)
    m, se = np.mean(data, axis=0), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_mean_and_CI(axes, mean, lb, ub, label, freqs, linestyle='-'):
    """ Plot mean and confidence boundaries.

        Args:
        axes: plt.axes
        mean: np.ndarray
        lb: np.ndarray
        ub: np.ndarray
        label: string
        freqs: list
        linestyle: string

        Returns: plt.axes

        """

    axes.fill_between(freqs, ub, lb, alpha=.25)
    axes.plot(freqs, mean, label=label, marker = 'o', linestyle=linestyle)

    return axes

def pretty_plot(axes, metric_name, plot_path, source=None):
    """ Set plot parameters and save plot to svg file.

    Args:
    axes: plt.axes
    metric_name: string
    plot_path: string

    """

    axes.legend(prop={'size': 15})
    axes.set_xscale('log')
    axes.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    axes.xaxis.set_major_formatter(mticker.ScalarFormatter())
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in axes.xaxis.get_minor_ticks():
        tick.label.set_fontsize(20)
    axes.grid(True,which="both",ls="--",c='gray')
    axes.set_xlabel('Frequency', fontsize=30)
    axes.set_ylabel(metric_name, fontsize=30)
    axes.set_ylim(0, 1)
    if source != None:
        axes.set_title(metric_name + ' at source position ' + str(source), fontsize=30)
        filename = metric_name + '_at_source_position_' + str(source) + ".svg"
    else:
        axes.set_title(metric_name, fontsize=30)
        filename = metric_name + ".svg"
    axes.figure.savefig(os.path.join(plot_path, filename))
    plt.close()

def plot_2D(data, filepath):
    """ Plot 2D data without white margins.

    Args:
    data: np.ndarray
    filepath: string

    """

    plt.figure()
    plt.imshow(data)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.NullLocator())
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

""" Keras utility functions """
# Code adopted from https://github.com/MathiasGruber/PConv-Keras

class PConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        """Set PConv2D parameters.

            Args:
            config: dict
            train_bn: boolean

        """

        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Define PConv2D layer's weights. Adapted from Keras _Conv() layer.

        Args:
        input_shape: list

        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Sound field kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='sf_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        """Define PConv2D layer's logic.

        Args:
        inputs: list=[K.tensor, K.tensor]

        Returns: list=[K.tensor, K.tensor]

        """

        # Both sound field and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [sf, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        sfs = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to sound field
        sf_output = K.conv2d(
            (sfs*masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each psoition in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize sound field output
        sf_output = sf_output * mask_ratio

        # Apply bias only to the sound field (if chosen to do so)
        if self.use_bias:
            sf_output = K.bias_add(
                sf_output,
                self.bias,
                data_format=self.data_format)

        # Apply activations on the sound field
        if self.activation is not None:
            sf_output = self.activation(sf_output)

        return [sf_output, mask_output]

    def compute_output_shape(self, input_shape):
        """Define PConv2D layer's shape transformation logic.

        Args:
        input_shape: list

        Returns: list

        """

        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]

