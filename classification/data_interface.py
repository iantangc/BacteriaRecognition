from __future__ import print_function, division
from keras.utils import Sequence
import numpy as np
import os
import gc
import sys
import datetime
import h5py

class ModelDataInterface(object):
    def __init__(self, data_file_path, dim_x = 32, dim_y = 32, dim_z = 32, n_classes = 10):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.data_file_path = data_file_path
        self.data_file = h5py.File(data_file_path, "a")

    @staticmethod
    def get_exploration_order(dataset_size, shuffle = False):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(dataset_size)
        if shuffle:
            np.random.shuffle(indexes)
        return indexes
    
    def _crop_pad_image(self, image_segment):
        # Center crop followed by balanced padding
        s = image_segment.shape
        if s[0] > self.dim_x or s[1] > self.dim_y:
            startx = s[1] // 2 - (self.dim_x // 2)
            starty = s[0] // 2 - (self.dim_y // 2)
            image_segment = image_segment[max(starty, 0): starty + self.dim_y, max(startx, 0) : startx + self.dim_x]
            s = image_segment.shape
        full_pad = ((self.dim_x - s[0]), (self.dim_y - s[1]))
        result = np.lib.pad(image_segment, (( full_pad[0] // 2, full_pad[0] - full_pad[0] // 2), ( full_pad[1] // 2, full_pad[1] - full_pad[1] // 2), (0, 0)), 'constant', constant_values = 127 )
        
        return result
    
    @staticmethod
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

    def get_file_metadata(self, dataset_partition):
        file_info_set = self.data_file[dataset_partition + "_file_info_set"]
        file_metadata = file_info_set[:]
        file_metadata = [(image_name.decode('UTF-8'), region_count) for image_name, region_count in file_metadata]
        return file_metadata

    def get_label_descriptions(self):
        label_description_set = self.data_file["label_description_set"]
        label_descriptions = label_description_set[:]
        label_descriptions = [(label_id, label_description.decode('UTF-8')) for label_id, label_description in label_descriptions]
        return label_descriptions
    
    class DataSequence(Sequence):
        
        def initialize_new_epoch(self):
            self.indexes = ModelDataInterface.get_exploration_order(self.dataset_size, self.shuffle)
            
            if not self.label_only:
                if self.data_augmentation_enabled:
                    self.image_modification = np.random.random((self.dataset_size))
                else:
                    self.image_modification = np.zeros((self.dataset_size))
        
        def __init__(self, data_interface_object, dataset_partition, dim_x, dim_y, dim_z, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False):
            print("Preparing sequence generator for '" + dataset_partition + "'")
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
            
            if not data_only:
                self.label_set = self.parent.data_file[dataset_partition + "_label_set"]
                dataset_size = self.label_set.shape[0]
                self.dataset_size = dataset_size
                if cached:
                    self.label_cache = np.empty((dataset_size), dtype = 'int')
                    for i in range(dataset_size):
                        self.label_cache[i] = self.label_set[i]


            if not label_only:
                self.image_set = self.parent.data_file[dataset_partition + "_image_set"]
                self.mask_set = self.parent.data_file[dataset_partition + "_mask_set"]
                self.metadata_set = self.parent.data_file[dataset_partition + "_metadata_set"]
                dataset_size = self.image_set.shape[0]
                self.dataset_size = dataset_size

                if cached:
                    self.data_cache = np.empty((dataset_size, self.dim_x, self.dim_y, self.dim_z), dtype = 'uint8')
                    for i in range(dataset_size):
                        shape = tuple(self.metadata_set[i][0])
                        image_segment = np.resize(self.image_set[i], shape)
                        standard_image = self.parent._crop_pad_image(image_segment)
                        self.data_cache[i, :, : ,:] = standard_image

                    
            self.total_batch_num = int(np.ceil(self.dataset_size / float(self.batch_size)))
            self.initialize_new_epoch()
            
        def __len__(self):
            return self.total_batch_num
        
        def on_epoch_end(self):
            gc.collect()
            self.initialize_new_epoch()

        def get_batch(self, batch_i):
            'Generates data of batch_size samples' 
            indexes = self.indexes
            label_only = self.label_only
            data_only = self.data_only
            cached = self.cached
            batch_size = self.batch_size
            batch_data_ids = indexes[batch_i * batch_size : (batch_i + 1) * batch_size]

            # X : (n_samples, v_size, v_size, v_size)
            # Initialization
            n_samples = len(batch_data_ids)

            if not label_only:
                X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
                if cached:
                    for i, data_id in enumerate(batch_data_ids):
                        standard_image = np.copy(self.data_cache[data_id])

                        processed_image = ModelDataInterface.modify_image(standard_image, self.image_modification[data_id]) / 255.0
                        X[i, :, :, :] = processed_image
                else:
                    for i, data_id in enumerate(batch_data_ids):
                        shape = tuple(self.metadata_set[data_id][0])
                        image_segment = np.resize(self.image_set[data_id], shape)
                        standard_image = self.parent._crop_pad_image(image_segment)

                        processed_image = ModelDataInterface.modify_image(standard_image, self.image_modification[data_id]) / 255.0
                        X[i, :, :, :] = processed_image

            if not data_only:
                y = np.empty((n_samples), dtype = int)
                if cached:
                    for i, data_id in enumerate(batch_data_ids):
                        y[i] = self.label_cache[data_id]
                else:
                    for i, data_id in enumerate(batch_data_ids):
                        y[i] = self.label_set[data_id]
            
            if data_only:
                return X
            
            if label_only:
                return y

            return X, self.parent._sparsify(y)

        def __getitem__(self, idx):
            return self.get_batch(idx)
        
    def get_sequence_generator(self, dataset_partition, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False):
        seq_gen = ModelDataInterface.DataSequence(self, dataset_partition, self.dim_x, self.dim_y, self.dim_z, data_only, label_only, data_augmentation_enabled, shuffle, batch_size, cached)
        return seq_gen
    
    
    def get_generator(self, dataset_partition, data_only = False, label_only = False, data_augmentation_enabled = False, shuffle = False, batch_size = 32, cached = False):
        
        # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        print("Preparing python generator for '" + dataset_partition + "'")
        if not data_only:
            label_set = self.data_file[dataset_partition + "_label_set"]
            dataset_size = label_set.shape[0]
            if cached:
                label_cache = np.empty((dataset_size), dtype = 'int')
                for i in range(dataset_size):
                    label_cache[i] = label_set[i]
        
        if not label_only:
            image_set = self.data_file[dataset_partition + "_image_set"]
            mask_set = self.data_file[dataset_partition + "_mask_set"]
            metadata_set = self.data_file[dataset_partition + "_metadata_set"]
            dataset_size = image_set.shape[0]
            if cached:
                data_cache = np.empty((dataset_size, self.dim_x, self.dim_y, self.dim_z), dtype = 'uint8')
                for i in range(dataset_size):
                    shape = tuple(metadata_set[i][0])
                    image_segment = np.resize(image_set[i], shape)
                    standard_image = self._crop_pad_image(image_segment)
                    data_cache[i, :, : ,:] = standard_image
                
        total_batch_num = int(np.ceil(dataset_size / float(batch_size)))
        
        def generate():
            'Generates batches of samples'
            
            # Infinite loop
            while 1:
                # Generate order of exploration of dataset
                
                indexes = ModelDataInterface.get_exploration_order(dataset_size, shuffle)
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
                        if cached:
                            for i, data_id in enumerate(batch_data_ids):
                                standard_image = np.copy(data_cache[data_id])

                                processed_image = ModelDataInterface.modify_image(standard_image, image_modification[data_id]) / 255.0
                                X[i, :, :, :] = processed_image
                        else:
                            for i, data_id in enumerate(batch_data_ids):
                                shape = tuple(metadata_set[data_id][0])
                                image_segment = np.resize(image_set[data_id], shape)
                                standard_image = self._crop_pad_image(image_segment)

                                processed_image = ModelDataInterface.modify_image(standard_image, image_modification[data_id]) / 255.0
                                X[i, :, :, :] = processed_image

                    if not data_only:
                        y = np.empty((n_samples), dtype = int)
                        if cached:
                            for i, data_id in enumerate(batch_data_ids):
                                y[i] = label_cache[data_id]
                        else:
                            for i, data_id in enumerate(batch_data_ids):
                                y[i] = label_set[data_id]
                    
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

class EvaluationDataInterface(object):
    dataset_exists_error_message_template = "Error encountered when creating {dataset_name}: dataset already exists."
    def __init__(self):
        pass

    def open_file(self, data_file_path):
        self.data_file_path = data_file_path
        self.data_file = h5py.File(data_file_path, "a")

    def write_evaluation_results_to_file(self, dataset_partition, table_text, count_cross_table, prob_label_cross_table, prob_pred_cross_table):
        
        if not dataset_partition + "table_text" in self.data_file:
            # print(table_text)
            self.data_file.create_dataset(dataset_partition + "table_text", data = np.array(table_text, dtype = "S256"), dtype = "S256")
        else:
            print(dataset_exists_error_message_template.format(dataset_name = dataset_partition + "count_cross_table"))

        if not dataset_partition + "count_cross_table" in self.data_file:
            self.data_file.create_dataset(dataset_partition + "count_cross_table", data = count_cross_table)
        else:
            print(dataset_exists_error_message_template.format(dataset_name = dataset_partition + "count_cross_table"))
        
        if not dataset_partition + "prob_label_cross_table" in self.data_file:
            self.data_file.create_dataset(dataset_partition + "prob_label_cross_table", data = prob_label_cross_table)
        else:
            print(dataset_exists_error_message_template.format(dataset_name = dataset_partition + "prob_label_cross_table"))
        
        if not dataset_partition + "prob_pred_cross_table" in self.data_file:
            self.data_file.create_dataset(dataset_partition + "prob_pred_cross_table", data = prob_pred_cross_table)
        else:
            print(dataset_exists_error_message_template.format(dataset_name = dataset_partition + "prob_pred_cross_table"))
        
    def read_evaluation_results(self, dataset_partition):
        table_text = self.data_file[dataset_partition + "table_text"]
        table_text = [x.decode('UTF-8') for x in table_text]
        
        count_cross_table = self.data_file[dataset_partition + "count_cross_table"]
        count_cross_table = np.array(count_cross_table)
        prob_label_cross_table = self.data_file[dataset_partition + "prob_label_cross_table"]
        prob_label_cross_table = np.array(prob_label_cross_table)
        prob_pred_cross_table = self.data_file[dataset_partition + "prob_pred_cross_table"]
        prob_pred_cross_table = np.array(prob_pred_cross_table)

        return table_text, count_cross_table, prob_label_cross_table, prob_pred_cross_table

    def write_label_descriptions(self, label_descriptions):
        index_name_tuple_type = np.dtype([
            ('label_id', np.uint32),
            ('label_description', 'S256')
        ])

        if not "label_description_set" in self.data_file:
            self.data_file.create_dataset("label_description_set", data = np.array(label_descriptions, dtype = index_name_tuple_type),  dtype = index_name_tuple_type, chunks = True)
        else:
            print(dataset_exists_error_message_template.format(dataset_name = "label_description_set"))

    def read_label_descriptions(self):
        label_description_set = self.data_file["label_description_set"]
        label_descriptions = label_description_set[:]
        label_descriptions = [(label_id, label_description.decode('UTF-8')) for label_id, label_description in label_descriptions]
        return label_descriptions
    

    def read_interpretation_results(self):
        headers = self.data_file["header_set"]
        headers = [x.decode('UTF-8') for x in headers]

        score_dictionary = {}
        score_set = self.data_file["score_set"]
        for k, v in score_set:
            score_dictionary[k.decode('UTF-8')] = v
        
        return headers, score_dictionary

    def close_file(self):
        self.data_file.close()

            
