import numpy as np
import os
import sys
import h5py

class H5DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, data_file_path, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True, n_classes = 10, data_augmentation = True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.data_file_path = data_file_path
        self.data_file = h5py.File(data_file_path, "a")
        self.data_augmentation = data_augmentation
    
    def __get_exploration_order(self, dataset_size):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(dataset_size)
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_data_ids):
        pass
    
    def __process_image(self, image_segment, image_modification):
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
        
        if image_modification < 0.25:
            pass
        elif image_modification < 0.5:
            result = np.flip(result, axis = 0)
        elif image_modification < 0.75:
            result = np.flip(result, axis = 1)
        else:
            result = np.flip(result, axis = 0)
            result = np.flip(result, axis = 1)
        return result

    def sparsify(self, y):
        'Returns labels in binary NumPy array'
        n_classes = self.n_classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
    
    def get_dataset_size(self, dataset_partition):
        label_set = self.data_file[dataset_partition + "_label_set"]
        return label_set.shape[0]
    
    def generate(self, dataset_partition):
        'Generates batches of samples'
        # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        
        image_set = self.data_file[dataset_partition + "_image_set"]
        mask_set = self.data_file[dataset_partition + "_mask_set"]
        label_set = self.data_file[dataset_partition + "_label_set"]
        metadata_set = self.data_file[dataset_partition + "_metadata_set"]
        
        dataset_size = image_set.shape[0]
        
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(dataset_size)
            if self.data_augmentation:
                image_modification = np.random.random((dataset_size))
            else:
                image_modification = np.zeros((dataset_size))
            # Generate batches
            total_batch_num = int(dataset_size / self.batch_size)
            for batch_i in range(total_batch_num):
                # Find list of IDs
                batch_data_ids = indexes[batch_i *self.batch_size : (batch_i + 1) * self.batch_size]

                'Generates data of batch_size samples' 
                # X : (n_samples, v_size, v_size, v_size)
                # Initialization
                n_samples = len(batch_data_ids)
                X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
                y = np.empty((n_samples), dtype = int)

                # Generate data
                for i, data_id in enumerate(batch_data_ids):
                    # Store volume
                    shape = tuple(metadata_set[data_id][0])
                    image_segment = np.resize(image_set[data_id], shape)
                    processed_image = self.__process_image(image_segment, image_modification[data_id])
                    
                    X[i, :, :, :] = processed_image / 255
                    y[i] = label_set[data_id]
                    
                yield X, self.sparsify(y)

    def generate_input_in_order(self, dataset_partition):
        'Generates batches of samples'
        # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        
        image_set = self.data_file[dataset_partition + "_image_set"]
        mask_set = self.data_file[dataset_partition + "_mask_set"]
        label_set = self.data_file[dataset_partition + "_label_set"]
        metadata_set = self.data_file[dataset_partition + "_metadata_set"]
        
        dataset_size = image_set.shape[0]
        
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = np.arange(dataset_size)

            # Generate batches
            total_batch_num = int(np.ceil(dataset_size / self.batch_size))
            for batch_i in range(total_batch_num):
                # Find list of IDs
                batch_data_ids = indexes[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

                'Generates data of batch_size samples' 
                # X : (n_samples, v_size, v_size, v_size)
                # Initialization
                n_samples = len(batch_data_ids)
                X = np.empty((n_samples, self.dim_x, self.dim_y, self.dim_z))
#                 y = np.empty((n_samples), dtype = int)

                # Generate data
                for i, data_id in enumerate(batch_data_ids):
                    # Store volume
                    shape = tuple(metadata_set[data_id][0])
                    image_segment = np.resize(image_set[data_id], shape)
                    processed_image = self.__process_image(image_segment)
                    
                    X[i, :, :, :] = processed_image / 255
#                     y[i] = label_set[data_id]
                    
#                 yield X, self.sparsify(y)
                yield X
    
    def generate_label_in_order(self, dataset_partition):
        'Generates batches of samples'
        # dataset_partition is a string specifying the partition to be used. i.e. 'training', 'dev' or 'testing'
        
        label_set = self.data_file[dataset_partition + "_label_set"]
        
        dataset_size = label_set.shape[0]
        
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = np.arange(dataset_size)

            # Generate batches
            total_batch_num = int(np.ceil(dataset_size / self.batch_size))
            for batch_i in range(total_batch_num):
                # Find list of IDs
                batch_data_ids = indexes[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]

                'Generates data of batch_size samples' 
                # X : (n_samples, v_size, v_size, v_size)
                # Initialization
                n_samples = len(batch_data_ids)
                
                y = np.empty((n_samples), dtype = int)

                # Generate data
                for i, data_id in enumerate(batch_data_ids):
                    y[i] = label_set[data_id]
                    
#                 yield X, self.sparsify(y)
                yield y

