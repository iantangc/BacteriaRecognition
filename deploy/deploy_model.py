import tensorflow as tf
import numpy as np
import io
import time
import json
import base64
from PIL import Image
from matplotlib import pyplot as plt
from collections import defaultdict
import visualization_utils as vis_util

class DeployedModel(object):
    def __init__(self, path_to_model, category_index):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.category_index = category_index
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    
    def predict_single(self, img_np):
        img_expanded = np.expand_dims(img_np, axis=0)  
        return self.predict_batch(img_expanded)

    def predict_batch(self, img_np):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_np})
        return {'boxes': boxes, 'scores': scores, 'classes': classes, 'num': num}
    
    def draw_visualize_prediction(self, img_np, boxes, classes, scores, threshold=0.1, fontsize=6, alpha=127):
        img_cp = img_np.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_cp,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            min_score_thresh=threshold,
            line_thickness=1,
            fontsize=fontsize,
            alpha=alpha)
        return img_cp