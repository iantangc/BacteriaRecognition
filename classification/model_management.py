from __future__ import print_function, division
import keras
from keras.layers import Dense, Add, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import os
import gc
import sys
import h5py
import datetime

# import pydot
# import graphviz
# from IPython.display import SVG

from model_specification import *
from data_interface import *
import learning_parameters

# %matplotlib inline

def instantiate_new_model(model_identifier = 'default'):
    # model identifier is a key within the model_structure dictionary
    model_parameters = learning_parameters.get_model_parameters()
    file_parameters = learning_parameters.get_file_parameters()
    training_parameters = learning_parameters.get_training_parameters()
    learning_rate_schedule = learning_parameters.learning_rate_schedule

    model, model_name = resnet_model(input_shape = model_parameters['input_shape'], num_output_classes = model_parameters['num_output_classes'], parameters = model_parameters['model_structure'][model_identifier])

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = model_parameters['optimizer'],
                  metrics = ['accuracy'])
    print(model_name)
    print(model.summary())
    # plot_model(model, to_file='ResNetModel.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

    if not os.path.isdir(file_parameters['save_dir']):
        os.makedirs(file_parameters['save_dir'])
    filepath = os.path.join(file_parameters['save_dir'], file_parameters['model_save_file_name'])
    
    model.save(filepath.format(epoch = 0))
    
    print("Model saved at " + filepath.format(epoch = 0))
    return filepath.format(epoch = 0)

def train_model(model_file_name, initial_epoch, num_epoch):

    model_parameters = learning_parameters.get_model_parameters()
    file_parameters = learning_parameters.get_file_parameters()
    training_parameters = learning_parameters.get_training_parameters()
    learning_rate_schedule = learning_parameters.learning_rate_schedule

    if not os.path.isdir(file_parameters['save_dir']):
        os.makedirs(file_parameters['save_dir'])
    filepath = os.path.join(file_parameters['save_dir'], file_parameters['model_save_file_name'])

    model = load_model(model_file_name)

    batch_size = training_parameters['batch_size']
    datagen = ModelDataInterface(file_parameters['data_file_path'], dim_x = model_parameters['input_shape'][0], 
        dim_y = model_parameters['input_shape'][1], dim_z = model_parameters['input_shape'][2], 
        n_classes = model_parameters['num_output_classes'])

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 period=file_parameters['file_save_period'],
                                 save_best_only=False)

    lr_scheduler = LearningRateScheduler(learning_parameters.learning_rate_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    print('Training starts')
    tick = datetime.datetime.now()

    # Fit the model on the batches generated by datagen.flow().
    if training_parameters['generator_option'] == 'sequence':
        training_gen = datagen.get_sequence_generator("training", data_only = False, label_only = False, data_augmentation_enabled = training_parameters['data_augmentation'], shuffle = training_parameters['shuffle'], batch_size = batch_size, cached = training_parameters['cached_training'])
        validation_gen = datagen.get_sequence_generator("dev", data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = batch_size, cached = False)
        num_workers = training_parameters['sequence_generator_workers']
    else:
        training_gen = datagen.get_generator("training", data_only = False, label_only = False, data_augmentation_enabled = training_parameters['data_augmentation'], shuffle = training_parameters['shuffle'], batch_size = batch_size, cached = training_parameters['cached_training'])
        validation_gen = datagen.get_generator("dev", data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = batch_size, cached = False)
        num_workers = 1
        
    
    model.fit_generator(
        generator = training_gen,
        steps_per_epoch = int(np.ceil(datagen.get_dataset_size("training") / float(batch_size))),
        validation_data = validation_gen,
        validation_steps = int(np.ceil(datagen.get_dataset_size("dev") / float(batch_size))),
        epochs = num_epoch + initial_epoch, 
        verbose = training_parameters['training_verbose_option'], 
        workers = num_workers,
        use_multiprocessing = True if num_workers > 1 else False,
        initial_epoch = initial_epoch,
        callbacks = callbacks)

    tock = datetime.datetime.now()
    diff_time = tock - tick
    print('Training Finished. Time taken:' + str(diff_time.total_seconds()))

    total_epoch = initial_epoch + num_epoch
    filepath = os.path.join(file_parameters['save_dir'], file_parameters['model_save_file_name']).format(epoch = total_epoch)
    if not os.path.isfile(filepath):
        print('Saving model at end of training')
        model.save(filepath)
        print('Model saved successfully')

def evaluate_model_overall(model_file_name):
    model_parameters = learning_parameters.get_model_parameters()
    file_parameters = learning_parameters.get_file_parameters()
    training_parameters = learning_parameters.get_training_parameters()

    model = load_model(model_file_name)
    
    batch_size = training_parameters['batch_size']

    datagen = ModelDataInterface(file_parameters['data_file_path'], dim_x = model_parameters['input_shape'][0], 
        dim_y = model_parameters['input_shape'][1], dim_z = model_parameters['input_shape'][2], 
        n_classes = model_parameters['num_output_classes'])

    print('Evaluation starts')
    scores = model.evaluate_generator(generator = datagen.get_generator("testing", data_only = False, label_only = False, 
                                                              data_augmentation_enabled = False, shuffle = False, batch_size = batch_size, cached = False),
                            steps = int(np.ceil(datagen.get_dataset_size("testing") / batch_size)), 
                            workers = 1,
                            use_multiprocessing = False) 
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def evaluate_model_detail(model_file_name, full_evalution = False, group_by_file = True, verbose = 1):
    model_parameters = learning_parameters.get_model_parameters()
    file_parameters = learning_parameters.get_file_parameters()
    training_parameters = learning_parameters.get_training_parameters()
    model = load_model(model_file_name)

    if full_evalution:
        file_used = file_parameters['full_data_file_path']
    else:
        file_used = file_parameters['data_file_path']

    batch_size = training_parameters['batch_size']
    datagen = ModelDataInterface(file_used, dim_x = model_parameters['input_shape'][0], 
        dim_y = model_parameters['input_shape'][1], dim_z = model_parameters['input_shape'][2], 
        n_classes = model_parameters['num_output_classes'])

    stage_texts = ['training', 'dev', 'testing']
    # stage_texts = ['dev', 'testing']
    n_classes = model_parameters['num_output_classes']

    save_time = datetime.datetime.now()
    save_time_str = save_time.strftime(file_parameters['evaluation_save_date_format'])
    evaluation_file_name = file_parameters['evaluation_save_file_name'].format(date = save_time_str)
    evaluation_file_interface = EvaluationDataInterface()
    evaluation_file_interface.open_file(evaluation_file_name)

    evaluation_file_interface.write_label_descriptions(label_descriptions = datagen.get_label_descriptions())
    
    print("Evaluating, and saving results to " + evaluation_file_name)
    for s, stage_text in enumerate(stage_texts):
        if verbose:
            print("\nEvalution for " + stage_text)

        stage_table_text = []
        stage_count_cross_table = np.zeros((0, n_classes, n_classes), dtype=int)
        stage_prob_label_cross_table = np.zeros((0, n_classes, n_classes), dtype=float)
        stage_prob_pred_cross_table = np.zeros((0, n_classes, n_classes), dtype=float)
        
        predictions = model.predict_generator(generator = datagen.get_sequence_generator(stage_text, data_only = True, label_only = False, 
                                                              data_augmentation_enabled = False, shuffle = False, batch_size = batch_size, cached = False),
                                workers = 1,
                                verbose = 0,
                                use_multiprocessing = False)
        predictions_flattened = np.argmax(predictions, axis = 1)

        label_gen = datagen.get_sequence_generator(stage_text, data_only = False, label_only = True, 
                                        data_augmentation_enabled = False, shuffle = False, batch_size = batch_size, cached = False)
        labels_flattened = np.array([], dtype = 'uint8')
        for i in range(len(label_gen)):
            labels_flattened = np.append(labels_flattened, label_gen.get_batch(i))

        count_cross_table = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(labels_flattened)):
            count_cross_table[labels_flattened[i], predictions_flattened[i]] += 1
        prob_label_cross_table, prob_pred_cross_table = cross_table_to_probability_table(count_cross_table)

        if verbose:
            print("Overall results")
            print("Count cross table")
            print(count_cross_table)
            print("Probability of given label")
            print(prob_label_cross_table)
            print("Probability of given prediction")
            print(prob_pred_cross_table)

        stage_table_text.append("overall")
        stage_count_cross_table = np.concatenate((stage_count_cross_table, np.reshape(count_cross_table, (1, n_classes, n_classes))), axis = 0)
        stage_prob_label_cross_table = np.concatenate((stage_prob_label_cross_table, np.reshape(prob_label_cross_table, (1, n_classes, n_classes))), axis = 0)
        stage_prob_pred_cross_table = np.concatenate((stage_prob_pred_cross_table, np.reshape(prob_pred_cross_table, (1, n_classes, n_classes))), axis = 0)

        if group_by_file:
            file_meta_data = datagen.get_file_metadata(stage_text)
            count_offset = 0
            
            for image_name, label_count in file_meta_data:
                count_cross_table = np.zeros((n_classes, n_classes), dtype=int)
                for i in range(label_count):
                    count_cross_table[labels_flattened[i + count_offset], predictions_flattened[i + count_offset]] += 1
                prob_label_cross_table, prob_pred_cross_table = cross_table_to_probability_table(count_cross_table)

                stage_table_text.append(image_name)
                stage_count_cross_table = np.concatenate((stage_count_cross_table, np.reshape(count_cross_table, (1, n_classes, n_classes))), axis = 0)
                stage_prob_label_cross_table = np.concatenate((stage_prob_label_cross_table, np.reshape(prob_label_cross_table, (1, n_classes, n_classes))), axis = 0)
                stage_prob_pred_cross_table = np.concatenate((stage_prob_pred_cross_table, np.reshape(prob_pred_cross_table, (1, n_classes, n_classes))), axis = 0)
                if verbose:
                    print("Image " +  image_name + " results")
                    print("Count cross table")
                    print(count_cross_table)
                    print("Probability of given label")
                    print(prob_label_cross_table)
                    print("Probability of given prediction")
                    print(prob_pred_cross_table)
                count_offset += label_count
        evaluation_file_interface.write_evaluation_results_to_file(stage_text, stage_table_text, stage_count_cross_table, stage_prob_label_cross_table, stage_prob_pred_cross_table)
    evaluation_file_interface.close_file()
    print("Evalution completed, all data written to " + evaluation_file_name)

def cross_table_to_probability_table(cross_table_2D):
    # prob_x: probability of C = y given R = x
    # prob_y: probability of R = x given C = y
    prob_x_cross_table = cross_table_2D.astype(float)
    cross_x_table_sum = cross_table_2D.sum(axis=-1, keepdims=True)
    
    for x in range(cross_table_2D.shape[-1]):
        if cross_x_table_sum[x, 0] == 0:
            for y in range(cross_table_2D.shape[-2]):
                # prob_x_cross_table[x, y] = 1 if x == y else 0
                prob_x_cross_table[x, y] = 0
        else:
            for y in range(cross_table_2D.shape[-2]):   
                prob_x_cross_table[x, y] /= cross_x_table_sum[x, 0]

    prob_y_cross_table = cross_table_2D.astype(float)
    cross_y_table_sum = cross_table_2D.sum(axis=-2, keepdims=True)
    
    for y in range(cross_table_2D.shape[-2]):
        if cross_y_table_sum[0, y] == 0:
            for x in range(cross_table_2D.shape[-1]):
                # prob_y_cross_table[x, y] = 1 if x == y else 0
                prob_y_cross_table[x, y] = 0
        else:
            for x in range(cross_table_2D.shape[-1]):   
                prob_y_cross_table[x, y] /= cross_y_table_sum[0, y]

                
    return prob_x_cross_table, prob_y_cross_table


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'init':
            if len(sys.argv) > 2:
                instantiate_new_model(sys.argv[2])
            else:
                print("Usage: python *.py init model_identifier")
        elif sys.argv[1] == 'train':
            if len(sys.argv) > 4:
                train_model(model_file_name = sys.argv[2], initial_epoch = int(sys.argv[3]), num_epoch = int(sys.argv[4]))
            else:
                print("Usage: python *.py train model_file_path initial_epoch num_epoch")
        elif sys.argv[1] == 'quicktest':
            if len(sys.argv) > 2:
                evaluate_model_overall(model_file_name = sys.argv[2])
            else:
                print("Usage: python *.py quicktest model_file_path")
        elif sys.argv[1] == 'eval':
            if len(sys.argv) > 4:
                evaluate_model_detail(model_file_name = sys.argv[2], full_evalution = sys.argv[3] == 'full', verbose = 1 if sys.argv[4] == 'v' else 0)
            else:
                print("Usage: python *.py eval model_file_path [full|part] [v|n]")
    else:
        print("Usage: python *.py [init|train|quicktest|eval]")