# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Training.py

from __future__ import division
import sys
sys.path.append('util')
import os
import util
import data
import sfun

def create_new_session(config):
    """ Creates a new folder to save all session artifacts to.
    Args:
    config: dict, session configuration parameters
    """

    if not os.path.exists('sessions'): os.mkdir('sessions')
    config['training']['session_dir'] = os.path.join('sessions', 'session_' + str(config['training']['session_id']))
    if not os.path.exists(config['training']['session_dir']): os.mkdir(config['training']['session_dir'])

def train(config_path):
    """ Trains a model

    Args:
    config_path: string, path to a config.json file
    """

    # Load configuration
    if not os.path.exists(config_path):
        print('Error: No configuration file present at specified path.')
        return

    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    # Create session directory
    if 'session_dir' not in config['training'] or os.path.exists(config['training']['session_dir']): create_new_session(config)

    model = sfun.SFUN(config)
    dataset = data.Dataset(config).load_dataset()

    train_set_generator = dataset.get_random_batch_generator('train')
    val_set_generator = dataset.get_random_batch_generator('val')

    model.fit_model(train_set_generator, config['training']['num_steps_train'], val_set_generator, config['training']['num_steps_val'],
                    config['training']['num_epochs'])

