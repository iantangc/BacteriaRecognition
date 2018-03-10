from __future__ import print_function, division
import os
import keras
import numpy as np

def get_training_parameters():
    params = {}
    params['data_augmentation'] = True
    params['batch_size'] = 32
    params['shuffle'] = True
    params['training_verbose_option'] = 1

    params['cached_training'] = False
    params['generator_option'] = 'standard' # 'standard' or 'sequence'
    params['sequence_generator_workers'] = 4

    return params

def learning_rate_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    lr *= np.power(1e-1, int(epoch / 80))

    # if epoch > 180:
    #     lr *= 0.5e-3
    # elif epoch > 160:
    #     lr *= 1e-3
    # elif epoch > 120:
    #     lr *= 1e-2
    # elif epoch > 80:
    #     lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_file_parameters():
    params = {}
    params['save_dir'] = os.path.join(os.getcwd(), 'saved_models')
    params['model_save_file_name'] = 'Test0131.{epoch:05d}.h5'
    params['file_save_period'] = 1

    params['dataset_dir_path'] = "../data/dataset/"
    params['evaluation_save_file_name'] = os.path.join(params['dataset_dir_path'], "Eval_{date}.h5") 
    params['evaluation_save_date_format'] = "%Y%m%d_%H%M"

    params['data_file_path'] = os.path.join(params['dataset_dir_path'], "dataset20180129Partial.h5")
    params['full_data_file_path'] = os.path.join(params['dataset_dir_path'], "dataset20180129Full.h5")
    return params

def get_model_parameters():
    params = {}
    # Training parameters
    
    params['input_shape'] = (224, 224, 3)
    params['num_output_classes'] = 9
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------

    # structure 1: wide 26 layers

    params['optimizer'] = keras.optimizers.Adam(lr=learning_rate_schedule(0), amsgrad=True)
    
    params['model_structure'] = {}
    params['model_structure']['Res23A'] = [
         [{'unit_type': 'std_conv', 'k': 7, 's': 2, 'f': 32, 'name': 'Input'}, 
          {'unit_type': 'max_pool', 'k': 3, 's': 2}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [32, 32, 128], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [32, 32, 128], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [64, 64, 256], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [64, 64, 256], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [128, 128, 512], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [128, 128, 512], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [128, 128, 512], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [128, 128, 512], 'name': 'res_id'}], 
         [{'unit_type': 'bn'},
          {'unit_type': 'activation', 'activation': 'relu'},
          {'unit_type': 'global_avg_pool'}, 
          {'unit_type': 'dense', 'num_units': params['num_output_classes'], 'activation': 'softmax', 'name': 'output_fc'}] 
        ]
    # structure 2: narrow 23 layers
    
    
    params['model_structure']['Res23B'] = [
         [{'unit_type': 'std_conv', 'k': 7, 's': 2, 'f': 32, 'name': 'Input'}, 
          {'unit_type': 'max_pool', 'k': 3, 's': 2}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [8, 8, 32], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [8, 8, 32], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [16, 16, 64], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [16, 16, 64], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [32, 32, 128], 'name': 'res_conv'}, 
          {'unit_type': 'res_block', 'k': 3, 's': 1, 'force_conv': False, 'fs': [32, 32, 128], 'name': 'res_id'}], 
         [{'unit_type': 'res_block', 'k': 3, 's': 2, 'force_conv': False, 'fs': [32, 32, 128], 'name': 'res_conv'}], 
         [{'unit_type': 'bn'},
          {'unit_type': 'activation', 'activation': 'relu'},
          {'unit_type': 'global_avg_pool'}, 
          {'unit_type': 'dense', 'num_units': params['num_output_classes'], 'activation': 'softmax', 'name': 'output_fc'}] 
        ]

    return params




