from __future__ import print_function, division
from keras.utils import Sequence
import numpy as np
import os
import gc
import sys
import h5py

class DataInterface(object):
    'Generates data for Keras'
    def __init__(self, data_file_path, dim_x = 32, dim_y = 32, dim_z = 32, n_classes = 10):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.data_file_path = data_file_path
        self.data_file = h5py.File(data_file_path, "a")
    
    def get_exploration_order(dataset_size, shuffle = False):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(dataset_size)
        if shuffle:
            np.random.shuffle(indexes)
        return indexes
    
    def _crop_pad_image(self, image_segment):
        # Center crop followed by balanced padding
        # image_modification is a random seed / fixed value for data augmentation
        s = image_segment.shape
        if s[0] > self.dim_x or s[1] > self.dim_y:
            startx = s[1] // 2 - (self.dim_x // 2)
            starty = s[0] // 2 - (self.dim_y // 2)
            image_segment = image_segment[max(starty, 0): starty + self.dim_y, max(startx, 0) : startx + self.dim_x]
            s = image_segment.shape
        full_pad = ((self.dim_x - s[0]), (self.dim_y - s[1]))
        result = np.lib.pad(image_segment, (( full_pad[0] // 2, full_pad[0] - full_pad[0] // 2), ( full_pad[1] // 2, full_pad[1] - full_pad[1] // 2), (0, 0)), 'constant', constant_values = 127 )
        
        return result
    
    def modify_image(image, image_modification_seed):
        if image_modification_seed < 0.25:
            pass
        elif image_modification_seed < 0.5:
            image = np.flip(image, axis = 0)
        elif image_modification_seed < 0.75:
            image = np.flip(image, axis = 1)
        else:
            image = np.flip(image, axis = 0)
            image = np.flip(image, axis = 1)
        return image
        
    def _sparsify(self, y):
        'Returns labels in binary NumPy array'
        n_classes = self.n_classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
    
    def get_dataset_size(self, dataset_partition):
        label_set = self.data_file[dataset_partition + "_label_set"]
        return label_set.shape[0]
    
    class DataSequence(Sequence):
        
        def initialize_new_epoch(self):
            self.indexes = DataInterface.get_exploration_order(self.dataset_size, self.shuffle)
            
            if not self.label_only:
                if self.data_augmentation_enabled:
                    self.image_modification = np.random.random((self.dataset_size))
                else:
                    self.image_modification = np.zeros((self.dataset_size))
        
        def __init__(self, data_interface_object, dataset_partition, dim_x, dim_y, dim_z, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False,):
            
            self.parent = data_interface_object
            self.dim_x = dim_x
            self.dim_y = dim_y
            self.dim_z = dim_z
            self.data_only = data_only
            self.label_only = label_only
            self.data_augmentation_enabled = data_augmentation_enabled
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.cached = cached
            

            
            if cached:
                self.data_cache = {}
            
            if not data_only:
                self.label_set = self.parent.data_file[dataset_partition + "_label_set"]
                self.dataset_size = self.label_set.shape[0]
                if cached:
                    self.data_cache['data'] = {}

            if not label_only:
                self.image_set = self.parent.data_file[dataset_partition + "_image_set"]
                self.mask_set = self.parent.data_file[dataset_partition + "_mask_set"]
                self.metadata_set = self.parent.data_file[dataset_partition + "_metadata_set"]
                self.dataset_size = self.image_set.shape[0]
                if cached:
                    self.data_cache['label'] = {}
                    
            self.total_batch_num = int(np.ceil(self.dataset_size / float(self.batch_size)))
            self.initialize_new_epoch()
            
        def __len__(self):
            return int(np.ceil(self.dataset_size / float(self.batch_size)))
        
        
        def on_epoch_end(self):
            gc.collect()
            self.initialize_new_epoch()
        
        def __getitem__(self, idx):
            'Generates data of batch_size samples' 
            batch_data_ids = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

            # X : (n_samples, v_size, v_size, v_size)
            # Initialization
            n_samples = len(batch_data_ids)

            if not self.label_only:
                X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
            if not self.data_only:
                y = np.empty((n_samples), dtype = int)

            # Generate data
            for i, data_id in enumerate(batch_data_ids):
                # Store volume
                if not self.label_only:
                    if not self.cached or data_id not in self.data_cache['data']:
                        shape = tuple(self.metadata_set[data_id][0])
                        image_segment = np.resize(self.image_set[data_id], shape)
                        standard_image = self.parent._crop_pad_image(image_segment)

                        if self.cached:
                            self.data_cache['data'][data_id] = np.copy(standard_image)
                    else:
                        standard_image = np.copy(self.data_cache['data'][data_id])

                    processed_image = DataInterface.modify_image(standard_image, self.image_modification[i]) / 255.0
                    X[i, :, :, :] = processed_image

                if not self.data_only:

                    if not self.cached or data_id not in self.data_cache['label']:
                        y[i] = self.label_set[data_id]
                        if self.cached:
                            self.data_cache['label'][data_id] = y[i]
                    else:
                        y[i] = self.data_cache['label'][data_id]

            if self.data_only:
                return X

            if self.label_only:
                return y

            return X, self.parent._sparsify(y)
        
    def get_sequence_generator(self, dataset_partition, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False):
        seq_gen = DataInterface.DataSequence(self, dataset_partition, self.dim_x, self.dim_y, self.dim_z, data_only, label_only, data_augmentation_enabled, shuffle, batch_size, cached)
        return seq_gen
    
    
    def get_generator(self, dataset_partition, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False):
        
        # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
    
        data_cache = {}
        
            
        if not data_only:
            label_set = self.data_file[dataset_partition + "_label_set"]
            dataset_size = label_set.shape[0]
            if cached:
                data_cache['data'] = {}
        
        if not label_only:
            image_set = self.data_file[dataset_partition + "_image_set"]
            mask_set = self.data_file[dataset_partition + "_mask_set"]
            metadata_set = self.data_file[dataset_partition + "_metadata_set"]
            dataset_size = image_set.shape[0]
            if cached:
                data_cache['label'] = {}
                
        total_batch_num = int(np.ceil(dataset_size / float(batch_size)))
        
        def generate():
            'Generates batches of samples'
            
            # Infinite loop
            while 1:
                # Generate order of exploration of dataset
                
                indexes = DataInterface.get_exploration_order(dataset_size, shuffle)
                if not label_only:
                    if data_augmentation_enabled:
                        image_modification = np.random.random((dataset_size))
                    else:
                        image_modification = np.zeros((dataset_size))
                # Generate batches
                for batch_i in range(total_batch_num):
                    # Find list of IDs
                    batch_data_ids = indexes[batch_i * batch_size : (batch_i + 1) * batch_size]

                    'Generates data of batch_size samples' 
                    # X : (n_samples, v_size, v_size, v_size)
                    # Initialization
                    n_samples = len(batch_data_ids)
                    
                    if not label_only:
                        X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
                    if not data_only:
                        y = np.empty((n_samples), dtype = int)

                    # Generate data
                    for i, data_id in enumerate(batch_data_ids):
                        # Store volume
                        
                        if not label_only:
                            if not cached or data_id not in data_cache['data']:
                                shape = tuple(metadata_set[data_id][0])
                                image_segment = np.resize(image_set[data_id], shape)
                                standard_image = self._crop_pad_image(image_segment)

                                if cached:
                                    data_cache['data'][data_id] = np.copy(standard_image)
                            else:
                                standard_image = np.copy(data_cache['data'][data_id])
                            
                            processed_image = DataInterface.modify_image(standard_image, image_modification[i]) / 255.0
                            X[i, :, :, :] = processed_image
                        
                        if not data_only:
                            
                            if not cached or data_id not in data_cache['label']:
                                y[i] = label_set[data_id]
                                if cached:
                                    data_cache['label'][data_id] = y[i]
                            else:
                                y[i] = data_cache['label'][data_id]
                            
                    if data_only:
                        yield X
                    
                    if label_only:
                        yield y
                    
                    if not (data_only or label_only):
                        yield X, self._sparsify(y)
                    
                    del X
                    del y
                    
                gc.collect()
                        
        return generate()
    


#     def generate_input_in_order(self, dataset_partition):
#         'Generates batches of samples'
#         # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        
#         image_set = self.data_file[dataset_partition + "_image_set"]
#         mask_set = self.data_file[dataset_partition + "_mask_set"]
#         label_set = self.data_file[dataset_partition + "_label_set"]
#         metadata_set = self.data_file[dataset_partition + "_metadata_set"]
        
#         dataset_size = image_set.shape[0]
        
#         # Infinite loop
#         while 1:
#             # Generate order of exploration of dataset
#             indexes = np.arange(dataset_size)

#             # Generate batches
#             total_batch_num = int(np.ceil(dataset_size / self.batch_size))
#             for batch_i in range(total_batch_num):
#                 # Find list of IDs
#                 batch_data_ids = indexes[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

#                 'Generates data of batch_size samples' 
#                 # X : (n_samples, v_size, v_size, v_size)
#                 # Initialization
#                 n_samples = len(batch_data_ids)
#                 X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
# #                 y = np.empty((n_samples), dtype = int)

#                 # Generate data
#                 for i, data_id in enumerate(batch_data_ids):
#                     # Store volume
#                     shape = tuple(metadata_set[data_id][0])
#                     image_segment = np.resize(image_set[data_id], shape)
#                     processed_image = self.__process_image(image_segment)
                    
#                     X[i, :, :, :] = processed_image / 255
# #                     y[i] = label_set[data_id]
                    
# #                 yield X, self._sparsify(y)
#                 yield X
    
#     def generate_label_in_order(self, dataset_partition):
#         'Generates batches of samples'
#         # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        
#         label_set = self.data_file[dataset_partition + "_label_set"]
        
#         dataset_size = label_set.shape[0]
        
#         # Infinite loop
#         while 1:
#             # Generate order of exploration of dataset
#             indexes = np.arange(dataset_size)

#             # Generate batches
#             total_batch_num = int(np.ceil(dataset_size / self.batch_size))
#             for batch_i in range(total_batch_num):
#                 # Find list of IDs
#                 batch_data_ids = indexes[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

#                 'Generates data of batch_size samples' 
#                 # X : (n_samples, v_size, v_size, v_size)
#                 # Initialization
#                 n_samples = len(batch_data_ids)
                
#                 y = np.empty((n_samples), dtype = int)

#                 # Generate data
#                 for i, data_id in enumerate(batch_data_ids):
#                     y[i] = label_set[data_id]
                    
# #                 yield X, self.__sparsify(y)
#                 yield y

