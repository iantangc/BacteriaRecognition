import os

def get_training_parameters():
    params = {}
    params['epochs'] = 1
    params['data_augmentation'] = True
    params['batch_size'] = 32

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
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_file_parameters():
    params = {}
    params['save_dir'] = os.path.join(os.getcwd(), 'saved_models')
    params['model_save_file_name'] = '%s.{epoch:03d}.h5' % "ResNet26"
    params['file_save_period'] = 2
    params['log_save_file_name'] = '%s_log.{epoch:03d}.npy' % "ResNet26"
    params['data_file_path'] = "../data/dataset/dataset20180128.h5"
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
    params['model_structure'] = [
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

    return params
