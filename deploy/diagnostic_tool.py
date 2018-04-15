import os
import tkinter as tk
import tkinter.ttk as ttk
import traceback
from tkinter import *
from tkinter import filedialog, font
from tkinter.scrolledtext import ScrolledText
import threading

import numpy as np
import PIL
from PIL import Image, ImageTk

from deploy_model import DeployedModel
from interpretation_util import *


class DiagnosticSystem(object):
    def __init__(self, master):
        self.master = master
        self.threads = []
        self.analyse_threads = []
        self.analyse_stop = False
        self.img_path_list = []
        self.img_dict = {}
        self.file_changed = False
        self.label_map = {
            1: {'id': 1, 'name': 'Lactobacillus'}, 
            2: {'id': 2, 'name': 'Gardnerella'}, 
            3: {'id': 3, 'name': 'Curved Rods'}, 
            4: {'id': 4, 'name': 'Coccus'}, 
            5: {'id': 5, 'name': 'Yeast'}, 
            6: {'id': 6, 'name': 'Noise'}}

       

        self.status_bar = Frame(self.master, width=300)
        self.status_bar.grid(row=0, column=0, columnspan=2)
    
        self.status_text = StringVar()
        self.status_label = Label(self.status_bar, textvariable=self.status_text, height=1, width=100, fg='black')
        self.status_label.grid(row=0, column=0, columnspan=2, pady=5, padx=1)
        self.set_status("Please load a machine learning model", 2)
        self.status_text_orig_color = self.status_label.cget("background")



        self.top_menu = Frame(self.master, width=200)
        self.top_menu.grid(row=1, column=0, columnspan=2)

        self.browse_model_btn = Button(self.top_menu, text='Browse Model File', height=1, width=18, fg='red', command=self._browse_model)
        self.browse_model_btn.grid(row=0, column=1, pady=5, padx=1)

        self.button_original_color = self.browse_model_btn.cget("background")
        self.helpBtn = Button(self.top_menu, text='Help', fg="blue", height=1, width=7, command=self._pop_help)
        self.helpBtn.grid(row=0, column=2, padx=(0,5))

        # Body: canvas (image) + label + control
        self.canvas_height = 450
        self.main_panel = Frame(self.master, width=600, height=self.canvas_height+50)
        self.main_panel.grid(row=2, column=0)

        # # image related
        self.current_img_text = Label(self.main_panel, width=50, height=1, text="Current Image : N/A")
        self.current_img_text.grid(row=0, column=0, columnspan=3)

        self.canvas = Canvas(self.main_panel, width=600, height=self.canvas_height, highlightthickness=2, relief=SUNKEN)
        self.canvas.grid(row=1, column=0, columnspan=3)

        self.results_text = tk.scrolledtext.ScrolledText(self.main_panel, wrap=tk.WORD, width=80,  height=4)
        self.results_text.configure(font=("Arial", 10, "normal"))
        self.results_text.grid(row=2, column=0, columnspan=3)


        self.toggle_img_btn = Button(self.main_panel, text='Toggle result', height=1, width=15, state=DISABLED, fg='black', command=self._toggle_img)
        self.toggle_img_btn.grid(row=3, column=0)
        self.save_img_btn = Button(self.main_panel, text='Save image', height=1, width=15, state=DISABLED, fg='black', command=self._save_img)
        self.save_img_btn.grid(row=3, column=1)
        self.redraw_result_btn = Button(self.main_panel, text='Redraw result', height=1, width=15, state=DISABLED, fg='black', command=self._redraw_img)
        self.redraw_result_btn.grid(row=3, column=2)

        self.right_panel = Frame(self.master, width=40, height=self.canvas_height+50)
        self.right_panel.grid(row=2, column=1)

        self.threhold_slider_text = Label(self.right_panel, text='Threshold for scroing')
        self.threhold_slider_text.grid(row=0, column=0)
        self.threhold_slider = Scale(self.right_panel, from_=0, to=100, length=100, orient=HORIZONTAL)
        self.threhold_slider.set(10)
        self.threhold_slider.grid(row=0, column=1, columnspan=2)

        self.iou_slider_text = Label(self.right_panel, text='Threshold for IoU (non-maximum suppression)', wraplength=150)
        self.iou_slider_text.grid(row=1, column=0)
        self.iou_slider = Scale(self.right_panel, from_=0, to=100, length=100, orient=HORIZONTAL)
        self.iou_slider.set(10)
        self.iou_slider.grid(row=1, column=1, columnspan=2)

        self.fontsize_slider_text = Label(self.right_panel, text='Drawing Fontsize')
        self.fontsize_slider_text.grid(row=2, column=0)
        self.fontsize_slider = Scale(self.right_panel, from_=0, to=100, length=100, orient=HORIZONTAL)
        self.fontsize_slider.set(20)
        self.fontsize_slider.grid(row=2, column=1, columnspan=2)

        self.img_listbox  = Listbox(self.right_panel, width=30,  height=20)
        self.img_listbox.grid(row=3, column=0, columnspan=3)
        self.img_listbox.bind('<<ListboxSelect>>', self._select_img)

        self.right_bottom_panel = Frame(self.right_panel, width=40)
        self.right_bottom_panel.grid(row=4, column=0, columnspan=3)
        self.browse_img_btn = Button(self.right_bottom_panel, text='Browse images', height=3, width=10, wraplength=50, state=DISABLED, fg='black', command=self._browse_img)
        self.browse_img_btn.grid(row=0, column=0, pady=5, padx=1)
        self.analyse_btn = Button(self.right_bottom_panel, text='Analyse selected image', height=3, width=10, wraplength=50, state=DISABLED, fg='black', command=self._analyse)
        self.analyse_btn.grid(row=0, column=1, pady=5, padx=1)
        self.remove_img_btn = Button(self.right_bottom_panel, text='Remove selected image', height=3, width=10, wraplength=50, state=DISABLED, fg='black', command=self._remove_images)
        self.remove_img_btn.grid(row=0, column=2, pady=5, padx=1)

        
        

    def set_status(self, message, message_type=0):
        if message_type == 0:
            self.status_label.config(bg=self.status_text_orig_color)
        elif message_type == 1:
            self.status_label.config(bg="skyblue")
        elif message_type == 2:
            self.status_label.config(bg="lightgreen")
        elif message_type == 3:
            self.status_label.config(bg="gold")
        elif message_type == 4:
            self.status_label.config(bg="red")
        self.status_text.set("Status: " + message)
        self.master.update()

    def _save_img(self):
        save_path = filedialog.asksaveasfilename()
        if len(save_path) == 0:
            return
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        if self.current_img_mode == 0:
            self.img_dict[self.current_img]['original_pillow'].save(save_path,"PNG")
        else:
            self.img_dict[self.current_img]['drawn_pillow'].save(save_path,"PNG")
        self.set_status("Image is successfully save to " + save_path, 1)

    def _toggle_img(self):
        if self.current_img_mode == 0 and self.img_dict[self.current_img]['result_available']:
            self.current_img_mode = 1
            self._show_img(self.img_dict[self.current_img]['drawn_pillow'])
        else:
            self.current_img_mode = 0
            self._show_img(self.img_dict[self.current_img]['original_pillow'])


    def _redraw_img(self):
        if self.img_dict[self.current_img]['result_available']:
            self.redraw_img(self.current_img)

    def _remove_images(self):
        index = int(self.img_listbox.curselection()[0])
        value = self.img_listbox.get(index)
        self.img_listbox.delete(index)
        del self.img_dict[value]
        self.remove_img_btn.configure(state=DISABLED)



    def _select_img(self, evt):
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.current_img = value
        self.current_img_text['text'] = "Current Image : " + self.current_img
        self.current_img_mode = 0
        self._show_img(self.img_dict[value]['original_show'])
        self.analyse_btn.configure(state=NORMAL)
        self.remove_img_btn.configure(state=NORMAL)
        if self.img_dict[value]['result_available'] == False:
            self.toggle_img_btn.configure(state=DISABLED)
            self.redraw_result_btn.configure(state=DISABLED)
        

    def _analyse(self):
        if len(self.analyse_threads) != 0:
            self.analyse_stop = True
            self.set_status("Stopping diagnosis job...", 3)
            return
        t = threading.Thread(target=self.start_analyse_current_image, args=(self.current_img,))
        self.analyse_threads.append(t)
        print(self.analyse_threads)
        t.start()
        self.analyse_btn.configure(text="Stop analyse", bg="red")
        

    def start_analyse_current_image(self, image_key):
        current_img = self.img_dict[image_key]
        target_w = 160
        target_h = 160

        combined_data = {'ymin': np.array([]), 'xmin': np.array([]), 'ymax': np.array([]), 'xmax': np.array([]),
                        'bbox': np.empty((0, 4)), 'class_label': np.array([], dtype=np.uint8), 'scores': np.array([])}
        
        scale=(160,160)
        shape=(8,6)
        
        label_num = 0
        h_pieces = current_img['width'] // target_w
        v_pieces = current_img['height'] // target_h
        for j in range(v_pieces):
            for i in range(h_pieces):
                ya = j * target_h
                yb = (j + 1) * target_h

                xa = i * target_w
                xb = (i + 1) * target_w
                crop = current_img['original_np'][ya: yb, xa: xb, :]
                # print("i j xa xb ya yb:", i, j, xa, xb, ya, yb)
                self.set_status("Getting the results for {} {} / {} : {:.2%}".format(image_key, (j * h_pieces + i), (h_pieces * v_pieces), (j * h_pieces + i) / (h_pieces * v_pieces)), 2)
                pred = self.model.predict_single(crop)
                
                offset_x, offset_y = xa, ya
                
                combined_data['ymin'] = np.append(combined_data['ymin'], (pred['boxes'][0, :, 0] * scale[1] + offset_y) / (scale[1] * shape[1]))
                combined_data['xmin'] = np.append(combined_data['xmin'], (pred['boxes'][0, :, 1] * scale[0] + offset_x) / (scale[0] * shape[0]))
                combined_data['ymax'] = np.append(combined_data['ymax'], (pred['boxes'][0, :, 2] * scale[1] + offset_y) / (scale[1] * shape[1]))
                combined_data['xmax'] = np.append(combined_data['xmax'], (pred['boxes'][0, :, 3] * scale[0] + offset_x) / (scale[0] * shape[0]))
                combined_data['bbox'] = np.concatenate( (combined_data['bbox'], pred['boxes'][0]) )
                combined_data['class_label'] = np.append(combined_data['class_label'], pred['classes'][0])
                combined_data['scores']= np.append(combined_data['scores'], pred['scores'][0])
                label_num += pred['num']
                if self.analyse_stop:
                    self.analyse_threads.pop()
                    if len(self.analyse_threads) == 0:
                        self.analyse_stop = False
                        self.set_status("Job stopped", 1)
                        self.analyse_btn.configure(text="Analyse selected image", bg=self.button_original_color)
                    return
                # cv2.imwrite(image_output_dir + image_name_format.format(base_name[0], xa, ya, base_name[1]), crop)
        self.set_status("Prediction completed", 1)
        perm = np.argsort(combined_data['scores'])[::-1]
        combined_data['ymin'] = combined_data['ymin'][perm]
        combined_data['xmin'] = combined_data['xmin'][perm]
        combined_data['ymax'] = combined_data['ymax'][perm]
        combined_data['xmax'] = combined_data['xmax'][perm]
        combined_data['class_label'] = np.array(combined_data['class_label'][perm], dtype=int)
        combined_data['scores'] = combined_data['scores'][perm]
        combined_data['bbox'] = np.array([list(box) for box in zip(combined_data['ymin'], combined_data['xmin'], combined_data['ymax'], combined_data['xmax'])], dtype=float)

        for key in combined_data:
            current_img[key] = combined_data[key]

        self.redraw_img(image_key)
        self.analyse_threads.pop()
        if len(self.analyse_threads) == 0:
            self.analyse_stop = False
            self.analyse_btn.configure(text="Analyse selected image", bg=self.button_original_color)
    
    def redraw_img(self, image_key):
        current_img = self.img_dict[image_key]
        threshold = int(self.threhold_slider.get()) / 100.0
        same_class_iou = int(self.iou_slider.get()) / 100.0
        diff_class_iou = 0.3
        fontsize = int(self.fontsize_slider.get())
        alpha = 127

        label_map = {1: 0, 2: 1, 3: 2, 4: -1, 5: -1, 6: -1}
        def customized_score_fn(score, label): 
            if label == 1:
                return 1 if score >= 0.8 else 0
            if label == 2:
                return 1 if score >= 0.8 else 0
            if label == 3:
                return 1 if score >= 0.1 else 0
            return 0
        
        self.set_status("Non-maximum suppression starts", 2)
        filtered_data = non_maximum_suppression_apply(current_img, threshold=threshold, same_class_iou=same_class_iou, diff_class_iou=diff_class_iou)
        
        self.set_status("Filtering finished, starts with interpretation", 2)
        bacteria_count, nugent_scores, interpretation = get_interpretation_overall(filtered_data, label_map, customized_score_fn)
        # print(bacteria_count, nugent_scores, interpretation)
        
        self.set_status("Drawing image", 2)
        new_img = self.model.draw_visualize_prediction(
            current_img['original_np'],
            filtered_data['bbox'], 
            filtered_data['class_label'], 
            filtered_data['scores'], 
            threshold=threshold, 
            fontsize=fontsize,
            alpha=alpha
            )
        self.set_status("Finished drawing", 0)
        self.results_text.delete('1.0', END)
        self.results_text.insert(INSERT, "Interpretation: {} Total Nugent Score: {}, \nNugent scores: Lactobacillus = {}, Gardnerella = {}, Curved Rods = {}\nBacteria count: Lactobacillus = {}, Gardnerella = {}, Curved Rods = {}".format(
            str(interpretation), str(nugent_scores[0]), str(nugent_scores[1]), str(nugent_scores[2]), str(nugent_scores[3]), str(bacteria_count[0]), str(bacteria_count[1]), str(bacteria_count[2])
        ))

        current_img['drawn_pillow'] = Image.fromarray(new_img)
        current_img['result_available'] = True
        self.curr_img_show = Image.fromarray(new_img)
        self.save_img_btn['state'] = NORMAL
        self._show_img(self.curr_img_show)
        self.current_img_mode = 1
        self.toggle_img_btn.configure(state=NORMAL)
        self.redraw_result_btn.configure(state=NORMAL)

       



    def _browse_img(self):
        # display config
        self.master.update()
        new_img_path_list = list(filedialog.askopenfilenames())
        if len(new_img_path_list) == 0:
            return
        t = threading.Thread(target=self._load_imgs, args=(new_img_path_list,))
        self.threads.append(t)
        t.start()
        

    def _load_imgs(self, new_img_path_list):
        for f in new_img_path_list:
            fname = os.path.basename(f)
            self.set_status("Loading image " + f, 1)
            self.img_dict[fname] = {}
            d = self.img_dict[fname]
            d['full_path'] = f
            d['original_pillow'] = Image.open(f)
            d['width'], d['height'] = d['original_pillow'].size
            if d['height'] != 960:
                d['original_pillow'] = d['original_pillow'].resize(( int(d['width'] / d['height'] * 960), 960), PIL.Image.BILINEAR)
                d['width'], d['height'] = ( int(d['width'] / d['height'] * 960), 960)
            d['original_np'] = np.array(d['original_pillow'].getdata()).reshape((d['height'], d['width'], 3)).astype(np.uint8)
            d['original_show'] = d['original_pillow'].resize(( int(d['width'] / d['height'] * self.canvas.winfo_height()), self.canvas.winfo_height()), PIL.Image.BILINEAR)
            d['result_available'] = False
            self.img_listbox.insert(END, fname)

        
        self.set_status("Image loading completed.", 0)
        self._show_img(self.img_dict[fname]['original_show'], True)
        self.img_listbox.activate(END)
        self.current_img_text['text'] = "Current Image : " + fname
        self.current_img = fname
        self.current_img_mode = 0
        self.analyse_btn.configure(state=NORMAL)


    def _show_img(self, img, resize=True):
        self.curr_img_show = img.resize(( int(img.size[0] / img.size[1] * self.canvas.winfo_height()), self.canvas_height), PIL.Image.BILINEAR)
        self.tkImg = ImageTk.PhotoImage(self.curr_img_show)
        # # canvas
        self.canvas.create_image(0,0, anchor="nw", image=self.tkImg)

    def _browse_model(self):
        self.master.update()
        self.model_path = filedialog.askopenfilename()
        if len(self.model_path) == 0:
            return

        fname, fext = os.path.splitext(self.model_path)
        if not fext == '.pb':
            self._pop_error("File type incompatible, please choose a .pb file.")
            return

        try:
            t = threading.Thread(target=self.load_model)
            self.threads.append(t)
            t.start()
        except:
            print("Error: unable to start thread")

    def load_model(self):
        self.set_status("Loading model", 3)
        self.model = DeployedModel(self.model_path, self.label_map)
        self.browse_img_btn['state'] = NORMAL
        self.master.title('BV Diagnostic System - Model: ' + self.model_path)
        self.browse_model_btn.configure(fg="black")
        self.set_status("Loading finished", 0)
    
    def _exit(self):
        self.master.quit()
        return

    def _pop_help(self):
        self.helpWin = Toplevel(width=200, height=150)
        self.helpWin.title('Help')
        text = 'The tool is designed for diagnosing Gardnerella Vaginosis:\n\n'
        text = text + '· Start by selecting a model file and loading a machine learnin model.\n Please wait for the model to load into the memory.\n\n'
        text = text + '· Then use the browse image function to load images to be diagnosed.\n You may select multiple files at once.\n\n'
        text = text + '· Select an image to be diagnosed in the list, then click the button "Ananslyse selected image"\n The diagnosis job will then start, you may check the process in the status bar.\n It is also possible to stop the diagnosis half-way, please allow around 20 seconds for the system to stop before restarting. \n\n'
        text = text + '· After diagnosis, a summary is provided in the textbox below the image, \n and the correponding labels are drwan on the image.\n\n'
        text = text + '. It is possible to tune the thresholds and visualization for recognition using the sliders at the right upper corner.\n Please note that it is possible to redraw the results without redoing the analysis by clicking the redraw result button\n\n'
        text = text + '. The annotated image can be exported to a PNG file by clicking the save image button \n The resutls could be toggled on and off using the Toggle result button\n\n'
        text = text + '. The data will be lost after closing the program, and temporary data can be erased using the remove image button.\n\n'
        text = text + '\n\nBacterial Vaginosis Diagnostic System V0.1 - 2018 Chi Ian Tang'
        text = text + '\n\nA short script from Google Tensorflow Research is modified and adopted in this tool.\n see https://github.com/tensorflow/models/tree/master/research/object_detection '

        label = Label(self.helpWin, text=text, justify=LEFT, wraplength=600, font='Helvetica 11')
        btn = Button(self.helpWin, text='Okay', width=16, height=1, command=self.helpWin.destroy)
        label.grid(row=0, column=0, padx=20, pady=20)
        btn.grid(row=1, column=0, padx=(0,20), pady=(0,20), sticky=E)

    def _pop_error(self, msg):
        self.errWin = Toplevel(width=100, height=70)
        self.errWin.title('Error')
        label = Label(self.errWin, text=msg, justify=LEFT, wraplength=600, font='Helvetica 11')
        btn = Button(self.errWin, text='Ok', width=16, height=1, command=self.errWin.destroy)
        label.grid(row=0, column=0, padx=20, pady=20)
        btn.grid(row=1, column=0, padx=(0,20), pady=(0,20), sticky=E)
        return

# def visualize_single_image(combined_data, original_img, threshold=0.1, scale=1.0/160.0, fontsize=12, line_thickness=3, alpha=128, max_boxes_to_draw=500, save_to_file=None):   
#     fig, ax = plt.subplots(figsize=(original_img.shape[1] * scale, original_img.shape[0] * scale))
# #     fig, ax = plt.subplots()
# #     img_cp = np.zeros((original_img.shape[0], original_img.shape[1], original_img.shape[2] + 1), dtype=original_img.dtype)
# #     img_cp[:,:,:-1] = original_img
#     ax.axis('off')
#     ax.imshow(vis.draw_visulizations_on_image(original_img, 
#                                               combined_data['bbox'], 
#                                               combined_data['class_label'], 
#                                               combined_data['scores'], 
#                                               threshold=threshold, 
#                                               fontsize=fontsize, 
#                                               line_thickness=line_thickness, 
#                                               alpha=alpha,
#                                               max_boxes_to_draw=max_boxes_to_draw), aspect='auto')
#     if save_to_file is not None:
#         plt.savefig(save_to_file, bbox_inches='tight')
#     plt.show()

def get_interpretation_overall(combined_data, label_map, scoring_fn):
    classes = ["Lactobacillus", "Gardnerella", "Curved Rods"]
    class_counts = [0] * len(classes)
    labels = combined_data['class_label']
    scores = combined_data['scores']
    for i in range(combined_data['label_num']):
        mapped_label = label_map[labels[i]]
        if mapped_label >= 0:
            class_counts[label_map[labels[i]]] += scoring_fn(scores[i], labels[i])
    nugent_scores = get_nugent_score(*tuple(class_counts))
    interp = get_nugent_score_interpretation_str(nugent_scores[0])
    return class_counts, nugent_scores, interp

def non_maximum_suppression_apply(combined_data, threshold, same_class_iou, diff_class_iou, iou_mode="min_area", file_info=""):
    keep = non_maximum_suppression(combined_data, threshold, same_class_iou, diff_class_iou)
    filtered_data = {
        'xmin': combined_data['xmin'][keep],
        'xmax': combined_data['xmax'][keep],
        'ymin': combined_data['ymin'][keep],
        'ymax': combined_data['ymax'][keep],
        'bbox': combined_data['bbox'][keep], 
        'scores': combined_data['scores'][keep], 
        'class_label': combined_data['class_label'][keep],
        'label_num': len(keep),
        'source_file': file_info
    }
    return filtered_data

def non_maximum_suppression(combined_data, threshold=0.01, same_class_iou=0.5, diff_class_iou=0.6, iou_mode="min_area"):
    # if there are no boxes, return an empty list
    if len(combined_data['xmin']) == 0:
        return []

    # initialize the list of picked indexes 
    keep_indeces = []
 
    # grab the coordinates of the bounding boxes
    xmins = combined_data['xmin']
    ymins = combined_data['ymin']
    xmaxs = combined_data['xmax']
    ymaxs = combined_data['ymax']
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (xmaxs - xmins) * (ymaxs - ymins)
    idxs = np.argsort(combined_data['scores'])
    subset = np.where(combined_data['scores'][idxs] >= threshold)
    idxs = idxs[subset]

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        keep_indeces.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        intersect_xmins = np.maximum(xmins[i], xmins[idxs[:last]])
        intersect_ymins = np.maximum(ymins[i], ymins[idxs[:last]])
        intersect_xmaxs = np.minimum(xmaxs[i], xmaxs[idxs[:last]])
        intersect_ymaxs = np.minimum(ymaxs[i], ymaxs[idxs[:last]])
        same_label = combined_data['class_label'][i] == combined_data['class_label'][idxs[:last]]
 
        # compute the width and height of the bounding box
        ws = np.maximum(0.0, intersect_xmaxs - intersect_xmins)
        hs = np.maximum(0.0, intersect_ymaxs - intersect_ymins)
 
        if iou_mode == "min_area":
            min_area = np.minimum(areas[i], areas[idxs[:last]])
            # compute the ratio of overlap
            iou = (ws * hs) / min_area
        else:
            iou = (ws * hs) / (areas[i] + areas[idxs[:last]] - ws * hs)
        
        
        # delete all indexes from the index list that have
        to_be_deleted = np.concatenate((np.where( ((same_label) & (iou > same_class_iou)) | ((np.logical_not(same_label)) & (iou > diff_class_iou)))[0],
                                        [last]
        ))
        idxs = np.delete(idxs, to_be_deleted)
        # idxs = np.delete(idxs, np.concatenate(([last], np.where( (iou > same_class_iou))[0])))

        #         print(len(idxs))
    # return only the bounding boxes that were picked using the
    # integer data type
    return np.array(keep_indeces, dtype=int)

if __name__ == "__main__":
    master = Tk()
    master.title('BV Diagnostic System')
    master.resizable(1,1)

    # create proposal box displayer
    diagsys = DiagnosticSystem(master)
    master.mainloop()
