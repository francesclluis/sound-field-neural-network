# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Inference.py

import sys
import os
import util
import sfun
import data
import copy
import numpy as np

sys.path.append('util')

def get_latest_checkpoint_path(session_dir):
    """ Returns the path of the most recent checkpoint in session_dir.

        Args:
        session_dir: string

        Returns: string

    """
    checkpoints_path = os.path.join(session_dir, 'checkpoints')
    if os.path.exists(checkpoints_path) and util.dir_contains_files(checkpoints_path):
        checkpoints = os.listdir(checkpoints_path)
        checkpoints.sort(key=lambda x: os.stat(os.path.join(checkpoints_path, x)).st_mtime)
        last_checkpoint = checkpoints[-1]
        return os.path.join(checkpoints_path, last_checkpoint)
    else:
        return ''


def get_results_dict():
    """ Get the dictionary to save results.

        Returns: dict

    """
    return {'name': [], 'num_file': [], 'xDim': [], 'yDim': [], 'm2': [], 'num_mics': [], 'num_comb': [], 'freq': [], 'NMSE': [], 'SSIM': [], 'pattern': [], 'p_real': [], 'p_predicted': [], 'p_previous': []}


def get_test_filenames(test_path):
    """ Get the .mat filenames given a folder path.

        Args:
        test_path: string

        Returns: string

    """
    filenames = [filename for filename in os.listdir(test_path) if filename.endswith('.mat')]
    return filenames


def reconstruct_soundfield(model, sf_sample, mask, factor, frequencies, filename, num_file, com_num,
                           results_dict):
    """ Reconstruct and evaluate sound field

        Args:
        model: keras model
        sf_sample: np.ndarray
        factor: int
        frequencies: list
        filename: string
        num_file: int
        com_num: int
        results_dict: dict



        Returns: dict

    """

    # Create one sample batch. Expand dims
    sf_sample = np.expand_dims(sf_sample, axis=0)
    sf_gt = copy.deepcopy(sf_sample)

    mask = np.expand_dims(mask, axis=0)
    mask_gt = copy.deepcopy(mask)

    # preprocessing
    irregular_sf, mask = util.preprocessing(factor, sf_sample, mask)

    #predict sound field
    pred_sf = model.predict([irregular_sf, mask])

    #measured observations. To use in postprocessing
    measured_sf = util.downsampling(factor, copy.deepcopy(sf_gt))
    measured_sf = util.apply_mask(measured_sf, mask_gt)

    #compute csv fields
    split_filename = filename[:-4].split('_')
    pattern = np.where(mask_gt[0, :, :, 0].flatten() == 1)[0]
    num_mic = len(pattern)

    for freq_num, freq in enumerate(frequencies):

        #Postprocessing
        reconstructed_sf_slice = util.postprocessing(pred_sf, measured_sf, freq_num, pattern, factor)

        #Compute Metrics
        reconstructed_sf_slice = util.postprocessing(pred_sf, measured_sf, freq_num, pattern, factor)
        nmse = util.compute_NMSE(sf_gt[0, :, :, freq_num], reconstructed_sf_slice)

        data_range = sf_gt[0, :, :, freq_num].max() - sf_gt[0, :, :, freq_num].min()
        ssim = util.compute_SSIM(sf_gt[0, :, :, freq_num].astype('float32'), reconstructed_sf_slice, data_range)

        average_pressure_real = util.compute_average_pressure(sf_gt[0, :, :, freq_num])
        average_pressure_predicted = util.compute_average_pressure(reconstructed_sf_slice)
        average_pressure_previous = util.compute_average_pressure(measured_sf[0, :, :, freq_num])

        #store results
        results_dict['freq'].append(freq)
        results_dict['name'].append(filename[:-4])
        results_dict['xDim'].append(split_filename[2])
        results_dict['yDim'].append(split_filename[3])
        results_dict['m2'].append(split_filename[4])
        results_dict['num_mics'].append(num_mic)
        results_dict['num_comb'].append(com_num)
        results_dict['num_file'].append(num_file)
        results_dict['pattern'].append(pattern)
        results_dict['NMSE'].append(nmse)
        results_dict['SSIM'].append(ssim)
        results_dict['p_real'].append(average_pressure_real)
        results_dict['p_predicted'].append(average_pressure_predicted)
        results_dict['p_previous'].append(average_pressure_previous)

    return results_dict


def real_data_evaluation(config_path):
    """ Evaluates a trained model on real data.

        Args:
        config_path: string

    """

    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('/')+1]

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:
        print('Error: No checkpoint found in same directory as configuration file.')
        return

    model = sfun.SFUN(config, train_bn=False)

    predict_path = os.path.join(session_dir, 'real_data_evaluation', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                   '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                   str(config['evaluation']['step_mics']))
    if not os.path.exists(predict_path): os.makedirs(predict_path)


    filepath = os.path.join(config['dataset']['path'], 'real_soundfields','RoomB_soundfield.mat')

    # Get Ground Truth
    soundfield_1 = util.load_RoomB_soundfield(filepath, 0)
    soundfield_2 = util.load_RoomB_soundfield(filepath, 1)


    frequencies = util.get_frequencies()

    results_dict = get_results_dict()

    print('\nEvaluating model in real sound fields...\n')

    for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
        for source_num, source in enumerate(['source_1', 'source_2']):

            mask_generator = data.MaskGenerator(config['dataset']['xSamples']//config['dataset']['factor'], config['dataset']['ySamples']//config['dataset']['factor'], len(frequencies), num_mics=num_mics, rand_seed=3)

            for com_num in range(config['evaluation']['num_comb']):
                print("\twith "+ str(num_mics) + " mics, pattern number " + str(com_num) + " and source position " + str(source_num))
                if source_num:
                    input_soundfield = copy.deepcopy(soundfield_2)
                else:
                    input_soundfield = copy.deepcopy(soundfield_1)

                mask = mask_generator.sample()
                filename = str(source_num) + '_d_4.159_6.459_26.862.mat'

                results_dict = reconstruct_soundfield(model, input_soundfield, mask, config['dataset']['factor'],
                                                      frequencies, filename, source_num, com_num, results_dict)


    print('\nWriting real room results...\n')
    util.write_results(os.path.join(predict_path, 'results_RoomB_Dataset.csv'), results_dict)

    print('Analysing and plotting results...')
    util.analyze_and_plot_real_results(os.path.join(predict_path, 'results_RoomB_Dataset.csv'), config)

    print('Evaluation completed!')


def simulated_data_evaluation(config_path):
    """ Evaluates a trained model on simulated data.

        Args:
        config_path: string

    """

    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('/')+1]

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:  # Model weights are loaded when creating the model object
        print('Error: No checkpoint found in same directory as configuration file.')
        return

    model = sfun.SFUN(config, train_bn=False)

    evaluation_path = os.path.join(session_dir, 'simulated_data_evaluation', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                  '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                  str(config['evaluation']['step_mics']))
    if not os.path.exists(evaluation_path): os.makedirs(evaluation_path)

    test_path = os.path.join(config['dataset']['path'], 'simulated_soundfields', 'test')
    filenames = get_test_filenames(test_path)

    frequencies = util.get_frequencies()

    for num_file, filename in enumerate(sorted(filenames)):
        print('\nEvaluating model in simulated room ' + str(num_file) + '...\n')
        aux_sound = util.load_soundfield(os.path.join(test_path, filename), frequencies)
        results_dict = get_results_dict()

        for num_mics in range(config['evaluation']['min_mics'], config['evaluation']['max_mics'], config['evaluation']['step_mics']):
            mask_generator = data.MaskGenerator(config['dataset']['xSamples']//config['dataset']['factor'], config['dataset']['ySamples']//config['dataset']['factor'], len(frequencies), num_mics=num_mics, rand_seed=3)

            for com_num in range(config['evaluation']['num_comb']):
                soundfield_input = copy.deepcopy(aux_sound)

                mask = mask_generator.sample()

                print("\twith "+ str(num_mics) + " mics and pattern number " + str(com_num))

                results_dict = reconstruct_soundfield(model, soundfield_input, mask, config['dataset']['factor'],
                                                      frequencies, filename, num_file, com_num, results_dict)

                com_num += 1

        filename = 'results_file_number_' + str(num_file) + '.csv'
        print('\nWriting simulated room ' + str(num_file) + ' results...\n')
        util.write_results(os.path.join(evaluation_path, filename), results_dict)

    print('Analysing and plotting results...')
    util.analyze_and_plot_simulated_results(evaluation_path, session_dir, config)

    print('Evaluation completed!')


def visualize(config_path):
    """ Plot predictions of trained model on real data.

        Args:
        config_path: string

    """

    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    frequencies = util.get_frequencies()

    session_dir = config_path[:config_path.rfind('/')+1]

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:
        print('Error: No checkpoint found in same directory as configuration file.')
        return

    model = sfun.SFUN(config, train_bn=False)

    visualization_path = os.path.join(session_dir, 'visualization')
    if not os.path.exists(visualization_path): os.makedirs(visualization_path)

    filepath = os.path.join(config['dataset']['path'], 'real_soundfields','RoomB_soundfield.mat')

    mask_generator = data.MaskGenerator(config['dataset']['xSamples']//config['dataset']['factor'], config['dataset']['ySamples']//config['dataset']['factor'], len(frequencies), num_mics=config['visualization']['num_mics'])

    # Get measured sound field
    sf_sample = util.load_RoomB_soundfield(filepath, config['visualization']['source'])
    sf_gt = np.expand_dims(copy.deepcopy(sf_sample), axis=0)
    initial_sf = np.expand_dims(sf_sample, axis=0)

    # Get mask samples
    mask = mask_generator.sample()
    mask = np.expand_dims(mask, axis=0)

    # preprocessing
    irregular_sf, mask = util.preprocessing(config['dataset']['factor'], initial_sf, mask)

    # Scale ground truth sound field
    sf_gt = util.scale(sf_gt)

    print('\nPlotting Ground Truth Sound Field Scaled...')
    for num_freq, freq in enumerate(frequencies):
        print('\tat frequency ' + str(freq))
        util.plot_2D(sf_gt[0, ..., num_freq], os.path.join(visualization_path, str(freq) + '_Hz_Ground_Truth.png'))

    print('\nPlotting Irregular Sound Field...')
    for num_freq, freq in enumerate(frequencies):
        print('\tat frequency ' + str(freq))
        util.plot_2D(irregular_sf[0, ..., num_freq], os.path.join(visualization_path, str(freq) + '_Hz_Irregular_SF.png'))

    print('\nPlotting Mask...')
    for num_freq, freq in enumerate(frequencies):
        print('\tat frequency ' + str(freq))
        util.plot_2D(mask[0, ..., num_freq], os.path.join(visualization_path, str(freq) + '_Hz_Mask.png'))

    pred_sf = model.predict([irregular_sf, mask])

    print('\nPlotting Predicted Sound Field...')
    for num_freq, freq in enumerate(frequencies):
        print('\tat frequency ' + str(freq))
        util.plot_2D(pred_sf[0, ..., num_freq], os.path.join(visualization_path, str(freq) + '_Hz_Pred_SF.png'))

