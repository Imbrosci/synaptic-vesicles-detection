# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:35:25 2020

Generate the GUI and to conduct image analysis, result visualization
and proof-reading.

@author: imbroscb
"""

import numpy as np
from matplotlib.figure import Figure
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
import seaborn as sns
from tkinter import filedialog
import xlsxwriter
import pandas as pd
from time import sleep
import os
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from scipy import ndimage
import PIL
from PIL import Image, ImageOps
from CNNs_GaussianNoiseAdder import MultiClass, MultiClassPost


class VesAnalysisGui(tk.Frame):

    def __init__(self, master=None):

        tk.Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):

        self.master.title("PRESYNAPTIC VESICLES DETECTION TOOL")
        self.pack(fill=tk.BOTH, expand=1)
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        run_analysis = tk.Menu(menu)
        display = tk.Menu(menu)

        menu.add_cascade(label="Analysis", menu=run_analysis)
        menu.add_cascade(label="Results_check", menu=display)

        run_analysis.add_command(label="Vesicles detection",
                                 command=self.start_analysis1)
        run_analysis.add_separator()
        run_analysis.add_command(label='Exit', command=self.exit_program)

        display.add_command(label='Display detection on image',
                            command=self.image_plus_results1)
        display.add_command(label='Display graphic results',
                            command=self.graphic_results)
        display.add_command(label='Manual correction',
                            command=self.manual_correction1)

    def start_analysis1(self):

        # delete the old window if one exists and create a new one
        try:
            self.canvas1.delete('all')
        except Exception:
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
        try:
            self.toolbar.pack_forget()
            self.canvas2.get_tk_widget().pack_forget()
            self.canvas2 = None
        except Exception:
            pass

        label1 = tk.Label(self.master, text='Enter experiment name')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 100, window=label1)

        self.experiment_name = tk.Entry(self.master)
        self.canvas1.create_window(400, 135, window=self.experiment_name)

        label2 = tk.Label(self.master,
                          text='Enter the pixel size (1 side!) in nanometer')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label2)

        self.pixel_size = tk.Entry(self.master)
        self.canvas1.create_window(400, 235, window=self.pixel_size)

        label3 = tk.Label(
            self.master,
            text='Do you want the estimation of the vesicles area? (y/n)')
        label3.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label3)

        self.ves_area_analysis = tk.Entry(self.master)
        self.canvas1.create_window(400, 345, window=self.ves_area_analysis)

        label4 = tk.Label(
            self.master, text='Select the folder where the images are located')
        label4.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 400, window=label4)

        button1 = tk.Button(self.master, text='Search folder',
                            command=self.start_analysis2,
                            font=('Helvetica', '10'))
        self.canvas1.create_window(400, 435, window=button1)

    def start_analysis2(self):

        # delete the old window and create a new one
        self.canvas1.delete('all')
        self.master.directory = filedialog.askdirectory()
        sleep(1)

        self.var = tk.StringVar()
        label1 = tk.Label(self.master, textvariable=self.var)
        label1.config(font=('helvetica', 14))

        self.canvas1.create_window(400, 300, window=label1)
        self.var.set('Analysis is starting...')
        self.master.update_idletasks()
        sleep(1)

        # load the models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        PATH = self.master.directory + '/' + 'model.pth'
        PATH_post = self.master.directory + '/' + 'model_post.pth'
        if torch.cuda.is_available():
            self.model = MultiClass(out=2).to(device)
            self.model.load_state_dict(torch.load(PATH))
            self.model_post = MultiClassPost(out=2).to(device)
            self.model_post.load_state_dict(torch.load(PATH_post))
        else:
            self.model = MultiClass(out=2)
            self.model.load_state_dict(torch.load(PATH, map_location=device))
            self.model_post = MultiClassPost(out=2)
            self.model_post.load_state_dict(torch.load(PATH_post,
                                                       map_location=device))
        self.model.eval()
        self.model_post.eval()

        # get the pixel size and check if the vesicle area is to be estimated
        self.pixel_size_final = float(self.pixel_size.get())
        ves_area_analysis = self.ves_area_analysis.get()
        if (ves_area_analysis != 'n') and (ves_area_analysis != 'y'):
            ves_area_analysis = 'n'

        # start to write the result file
        image_dir = self.master.directory
        excel_name = self.experiment_name.get() + '.xlsx'
        book = xlsxwriter.Workbook(image_dir + '/' + excel_name)
        main_result_sheet = book.add_worksheet('Summary results')
        main_result_sheet.write(0, 0, 'Image name')
        main_result_sheet.write(0, 1, 'Vesicles count')

        self.counter = 1
        for file in os.listdir(image_dir):
            # check if the file is an image and if so open and process it
            extention = Path(file).suffix
            file_name = Path(file).stem
            if (extention == 'xlsx') or (extention == 'xls'):
                continue
            elif (extention == 'pth') or (extention == ''):
                continue
            elif ('_mask' in file_name):
                continue
            try:
                self.img_to_analyse = Image.open(image_dir + '/' + file)
            except PIL.UnidentifiedImageError:
                continue
            self.var.set('Processing image number: ' + str(self.counter))
            self.master.update_idletasks()
            sheet = book.add_worksheet(file_name)
            sleep(1)

            # process the image
            self.sliding_detection()
            if ves_area_analysis == 'y':
                self.mask.save(image_dir + '/' + file_name + '_mask.tif')

            # get the x,y coordinates of the detected vesicles
            if len(self.coordinates) > 0:
                x, y = zip(*self.coordinates)

                # get the nearest neighbor distances (nnds) between vesicles
                X = np.array([x, y]).T
                euc_distances = euclidean_distances(X, X)
                euc_distances[np.where(euc_distances == 0.0)] = 10000
                min_distances = []
                for i in range(euc_distances.shape[0]):
                    temp = min(euc_distances[i, :])
                    min_distances.append(temp * self.pixel_size_final)

                # fill the result file
                for i, e in enumerate(self.coordinates):
                    sheet.write(i + 1, 0, e[0])
                    sheet.write(i + 1, 1, e[1])
                    sheet.write(i + 1, 2, min_distances[i])
                    if ves_area_analysis == 'y':
                        sheet.write(i + 1, 3, self.area[i])
            sheet.write(0, 0, 'x_values')
            sheet.write(0, 1, 'y_values')
            sheet.write(0, 2, 'Distance to nearest vesicle (nm)')
            if ves_area_analysis == 'y':
                sheet.write(0, 3, 'Area (nm²)')
            main_result_sheet.write(self.counter, 0, file_name)
            main_result_sheet.write(self.counter, 1, len(self.coordinates))
            self.counter += 1

        book.close()
        self.var.set('Done!')
        print('Done!')
        self.master.update_idletasks()

    def exit_program(self):

        self.master.quit()
        self.master.destroy()

    def image_plus_results1(self):

        # delete old windows / images, if any, and create a new windo
        try:
            self.canvas1.delete('all')
        except Exception:
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
        try:
            self.toolbar.pack_forget()
            self.canvas2.get_tk_widget().pack_forget()
            self.canvas2 = None
        except Exception:
            pass

        label1 = tk.Label(
            self.master,
            text=(
                'Enter the name of the excel file where the results are stored'
                ))
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label1)

        self.results_to_use = tk.Entry(self.master)
        self.canvas1.create_window(400, 250, window=self.results_to_use)

        label2 = tk.Label(self.master, text='Select one analysed image')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2)

        button1 = tk.Button(self.master, text='Search file',
                            command=self.image_plus_results2,
                            font=('Helvetica', '10'))
        self.canvas1.create_window(400, 350, window=button1)

    def image_plus_results2(self):

        self.canvas1.destroy()

        # get the image path and the possible mask path
        self.master.filepath = filedialog.askopenfilename(
            initialdir="/", title="Select file")
        image_path = self.master.filepath
        if image_path[-9:] == '_mask.tif':
            image_path = image_path[:-9] + '.tif'

        splitted = image_path.split('.')
        mask_path = image_path.split('.')[0] + '_mask.tif'

        if len(splitted) > 2:
            mask_path = image_path.split('.')[0]
            for s in range(len(splitted) - 2):
                mask_path = mask_path + '.' + image_path.split('.')[s+1]
            mask_path = mask_path + '_mask.tif'

        # try to open the image and the mask
        try:
            img = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            print('The selected file is not identified as an image')
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
            label1 = tk.Label(
                self.master,
                text='The selected file is not identified as an image')
            label1.config(font=('helvetica', 12))
            self.canvas1.create_window(400, 250, window=label1)
            button1 = tk.Button(self.master, text='Click here to try again',
                                command=self.image_plus_results1,
                                font=('Helvetica', '10'))
            self.canvas1.create_window(400, 350, window=button1)
        try:
            mask = Image.open(mask_path)
        except FileNotFoundError:
            mask = 'Not found'

        # search the sheet where the results of the selected image are stored
        list_path = image_path.split('/')
        self.sheet_name = Path(image_path).stem
        result_dir = ''
        for i in range(len(list_path) - 1):
            result_dir = result_dir + list_path[i] + '/'

        # try to read the result file
        error_free = 1
        if (self.results_to_use.get()[-4:] == 'xlsx') or (
                self.results_to_use.get()[-3:] == 'xls'):
            try:
                excel_filename = self.results_to_use.get()
                self.xls = pd.ExcelFile(result_dir + excel_filename)
            except FileNotFoundError:
                error_free = 0
                print('Result file not found')
                self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                         relief='raised')
                self.canvas1.pack()
                label1 = tk.Label(self.master, text='Result file not found')
                label1.config(font=('helvetica', 12))
                self.canvas1.create_window(400, 250, window=label1)
                button1 = tk.Button(self.master,
                                    text='Click here to try again',
                                    command=self.image_plus_results1,
                                    font=('Helvetica', '10'))
                self.canvas1.create_window(400, 350, window=button1)
        else:
            try:
                excel_filename = self.results_to_use.get() + '.xlsx'
                self.xls = pd.ExcelFile(result_dir + excel_filename)
            except FileNotFoundError:
                try:
                    excel_filename = self.results_to_use.get() + '.xls'
                    self.xls = pd.ExcelFile(result_dir + excel_filename)
                except FileNotFoundError:
                    error_free = 0
                    print('Result file not found')
                    self.canvas1 = tk.Canvas(self.master, width=800,
                                             height=600, relief='raised')
                    self.canvas1.pack()
                    label1 = tk.Label(self.master,
                                      text='Result file not found')
                    label1.config(font=('helvetica', 12))
                    self.canvas1.create_window(400, 250, window=label1)
                    button1 = tk.Button(self.master,
                                        text='Click here to try again',
                                        command=self.image_plus_results1,
                                        font=('Helvetica', '10'))
                    self.canvas1.create_window(400, 350, window=button1)

        if error_free == 1:
            try:
                df_labels = pd.read_excel(self.xls, self.sheet_name, header=0,
                                          engine='openpyxl')
            except ValueError:
                error_free = 0
                print('Analysis of the selected image not found')
                self.canvas1 = tk.Canvas(self.master, width=800,
                                         height=600,  relief='raised')
                self.canvas1.pack()
                label1 = tk.Label(
                    self.master,
                    text='Analysis of the selected image not found')
                label1.config(font=('helvetica', 12))
                self.canvas1.create_window(400, 250, window=label1)

                button1 = tk.Button(self.master,
                                    text='Click here to try again',
                                    command=self.image_plus_results1,
                                    font=('Helvetica', '10'))
                self.canvas1.create_window(400, 350, window=button1)

        if error_free == 1:
            # get the coordinates of the vesicles
            x = []
            y = []
            for idx, row in df_labels.iterrows():
                x.append(row['x_values'])
                y.append(row['y_values'])
            x = np.array(x)
            y = np.array(y)

            # plot image and labels
            self.fig = Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.imshow(img, cmap='gray')
            if mask != 'Not found':
                self.ax.imshow(mask)
            self.ax.scatter(x, y, c='white', s=4)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas2 = FigureCanvasTkAgg(self.fig, master=self.master)
            self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas2,
                                                      self.master)
            self.toolbar.update()
            self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def graphic_results(self):

        # delete old windows or images, if any, and create a new one
        try:
            self.canvas1.delete('all')
        except Exception:
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
        try:
            self.toolbar.pack_forget()
            self.canvas2.get_tk_widget().pack_forget()
            self.canvas2 = None
        except AttributeError:
            pass

        label1 = tk.Label(
            self.master,
            text='Select the folder where the results file/s is/are located')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label1)

        button1 = tk.Button(self.master, text='Search', command=self.plot,
                            font=('Helvetica', '10'))
        self.canvas1.create_window(400, 300, window=button1)

    def plot(self):

        self.canvas1.destroy()
        # for each image plot vesicle counts, mean nnd and, if applicable, the
        # mean estimated area. Each result file is considered an experiment.
        self.master.result_dir = filedialog.askdirectory()
        result_dir = self.master.result_dir
        experiments_list = []
        for file in os.listdir(result_dir):
            if file[-4:] == 'xlsx':
                experiments_list.append(file)
            elif file[-3:] == 'xls':
                experiments_list.append(file)
            else:
                continue
        # check if experiments exist
        if len(experiments_list) == 0:
            print('There is no result file in the selected folder')
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
            label1 = tk.Label(
                self.master,
                text='There is no result file in the selected folder')
            label1.config(font=('helvetica', 12))
            self.canvas1.create_window(400, 250, window=label1)
            button1 = tk.Button(self.master, text='Click here to try again',
                                command=self.graphic_results,
                                font=('Helvetica', '10'))
            self.canvas1.create_window(400, 350, window=button1)

        # load the results from the result files
        ves_count = []
        result_filename = []
        average_min_distance = []
        average_area = []

        # iterate over the experiments
        for experiment_name in experiments_list:
            xls = pd.ExcelFile(result_dir + '/' + experiment_name)
            temp = pd.read_excel(xls, 'Summary results', header=0,
                                 engine='openpyxl')
            temp = temp['Vesicles count']
            ves_count.append(np.array(temp))
            if experiment_name[-4:] == 'xlsx':
                result_filename.append(experiment_name[:-5])
            elif experiment_name[-3:] == 'xls':
                result_filename.append(experiment_name[:-4])
            sheet_list = xls.sheet_names
            av_dist_per_exp = np.zeros((len(sheet_list)-1))
            av_area_per_exp = np.zeros((len(sheet_list)-1))

            # iterate over the sheets
            for i in range(len(sheet_list) - 1):
                temp = pd.read_excel(xls, sheet_list[i+1], header=0,
                                     engine='openpyxl')
                temp_nn = temp['Distance to nearest vesicle (nm)']
                av_dist_per_exp[i] = np.mean(np.array(temp_nn))
                try:
                    temp_area = temp['Area (nm²)']
                except KeyError:
                    temp_area = pd.Series(dtype='object')
                if not temp_area.empty:
                    av_area_per_exp[i] = np.mean(np.array(temp_area))

            average_min_distance.append(av_dist_per_exp)
            if not temp_area.empty:
                average_area.append(av_area_per_exp)

        # organize data in dataframes for plotting with sns
        tuple1_list = []
        tuple2_list = []
        tuple3_list = []
        i = 0
        # iterate over the experiments
        for exp_name in result_filename:
            for j in range(len(ves_count[i])):
                tuple1_list.append((exp_name, ves_count[i][j]))
                tuple2_list.append((exp_name, average_min_distance[i][j]))
                if (len(average_area) > 0) and (
                        len(average_area) == len(average_min_distance)):
                    tuple3_list.append((exp_name, average_area[i][j]))
            i += 1

        # create the dataframes
        df_data1 = pd.DataFrame(tuple1_list,
                                columns=['Experiment', 'Vesicles count'])
        df_data2 = pd.DataFrame(
            tuple2_list,
            columns=['Experiment', 'Nearest neighbor distance (nm)'])

        if len(tuple3_list) > 0:
            df_data3 = pd.DataFrame(tuple3_list,
                                    columns=['Experiment', 'Area (nm²)'])
            columns = 3
        else:
            columns = 2

        # create the figure and plot the graphs
        fig = Figure(figsize=(10, 20))
        ax1 = fig.add_subplot(1, columns, 1)
        a = sns.swarmplot(x='Experiment', y='Vesicles count', data=df_data1,
                          ax=ax1)
        a.set_xlabel('Name of experiment', fontsize=12, position=(9, 0.3))
        a.set_ylabel('Vesicles count', fontsize=12)
        a.tick_params(labelsize=12)
        ax2 = fig.add_subplot(1, columns, 2)
        b = sns.swarmplot(x='Experiment', y='Nearest neighbor distance (nm)',
                          data=df_data2, ax=ax2)
        b.set_xlabel('Name of experiment', fontsize=12, position=(9, 3.6))
        b.set_ylabel('Mean nearest neighbor (nm)', fontsize=12)
        b.tick_params(labelsize=12)
        if columns == 3:
            ax3 = fig.add_subplot(1, columns, 3)
            c = sns.swarmplot(x='Experiment', y='Area (nm²)', data=df_data3,
                              ax=ax3)
            c.set_xlabel('Name of experiment', fontsize=12, position=(9, 6.9))
            c.set_ylabel('Mean area (nm²)', fontsize=12)
            c.tick_params(labelsize=12)

        self.canvas2 = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def sliding_detection(self):

        # resize the image and transform it from color to gray if necessary
        shape0 = self.img_to_analyse.size[0]
        shape1 = self.img_to_analyse.size[1]
        img = self.img_to_analyse.resize((
            int(self.img_to_analyse.size[0] * self.pixel_size_final / 2.27),
            int(self.img_to_analyse.size[1] * self.pixel_size_final / 2.27)))
        np_img = np.array(img)
        if len(np_img.shape) > 2:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

        # preparation for feeding first classifier
        sliding_size = 4
        window_size = 40

        # put padding on image
        np_img_padded = np.zeros((np_img.shape[0] + 40, np_img.shape[1] + 40))
        np_img_padded[20:np_img.shape[0] + 20,
                      20:np_img.shape[1] + 20] = np_img
        p_map = np.zeros((int(np_img.shape[0] / sliding_size),
                          int(np_img.shape[1] / sliding_size)))
        print('Processing image number: {:.0f}'.format(self.counter))

        # iterate over image.shape[1] in steps of size == sliding_size
        for x in range(0, np_img.shape[1], sliding_size):
            self.var.set('Processing image number: ' + str(self.counter))

        # iterate over image.shape[0] in steps of size == sliding_size
            for y in range(0, np_img.shape[0], sliding_size):
                snapshot = np_img_padded[y + 20:y + window_size + 20,
                                         x + 20:x + window_size + 20]

                if (snapshot.shape[0] != window_size) or (
                        snapshot.shape[1] != window_size):
                    continue
                snapshot = snapshot.reshape(1, snapshot.shape[0],
                                            snapshot.shape[1])
                if np.max(snapshot) != np.min(snapshot):
                    snapshot = (snapshot - np.min(snapshot)) / (
                        np.max(snapshot) - np.min(snapshot))
                snapshot = (snapshot - 0.5) / 0.5
                snapshot = torch.from_numpy(snapshot)
                snapshot = snapshot.unsqueeze(0)

                # feed images patches into the first classifier
                if torch.cuda.is_available():
                    output = self.model.forward(snapshot.float().cuda())
                    valuemax, preds = torch.max(output, 1)
                    valuemin, _ = torch.min(output, 1)
                    valuemax = valuemax.cpu()
                    valuemin = valuemin.cpu()
                    preds = preds.cpu()
                else:
                    output = self.model.forward(snapshot.float())
                    valuemax, preds = torch.max(output, 1)
                    valuemin, _ = torch.min(output, 1)
                if preds == 1:
                    valuemax = valuemax.data.numpy()
                    valuemin = valuemin.data.numpy()
                    pvalue = np.exp(valuemax) / (np.exp(valuemax) + np.exp(
                        valuemin))
                    p_map[int((y + 20) / sliding_size),
                          int((x + 20) / sliding_size)] = pvalue

        # generate proc_pmap resizing and blurring p_map
        proc_pmap = cv2.resize(p_map, (np_img.shape[1], np_img.shape[0]))
        proc_pmap = cv2.blur(proc_pmap, (3, 3))

        # rescale proc_pmap values (0-255)
        if np.max(proc_pmap) > 0:
            proc_pmap = (proc_pmap / (np.max(proc_pmap))) * 255

        # set a threshold for proc_map (below 20% of 255, pixel=0)
        for xx in range(proc_pmap.shape[0]):
            for yy in range(proc_pmap.shape[1]):
                if proc_pmap[xx, yy] < 255 / 100 * 20:
                    proc_pmap[xx, yy] = 0

        # get objects with connected-component labelling algorithm
        labelarray, counts = ndimage.measurements.label(proc_pmap)
        x_labels = []
        y_labels = []

        # iterate over the found objects (labelarray)
        for i in range(counts):
            x, y = np.where(labelarray == i + 1)
            temp = []
            if type(x) == int:
                temp.append(x, y)
            else:
                for j in range(len(x)):
                    temp.append((x[j], y[j]))
            try:
                euc_distances = euclidean_distances(temp, temp)
            except MemoryError:
                break

            # check what is the most likely number of vesicles in each object
            # this will define the number of clusters to use in Kmeans
            max_distance = np.max(euc_distances)
            numb_clusters = 1

            # if the max distance in an object > 25, check for peaks
            if max_distance > 25:
                peaks = []
                for j in range(len(x)):
                    temp_peak = proc_pmap[x[j], y[j]]
                    temp_peak_idx = (x[j], y[j])
                    challenge_peak = np.zeros((8))
                    gapx = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
                    gapy = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

                    # look around temp_peak to check if it is a peak
                    for g in range(8):
                        challenge_peak[g] = proc_pmap[x[j] + gapx[g],
                                                      y[j] + gapy[g]]
                    if (np.max(challenge_peak) <= temp_peak):
                        peaks.append(temp_peak_idx)

                # calculate the distance between the peaks,
                # the number peaks with distance > 15 will define n_clusters
                if len(peaks) > 1:
                    euc_distances = euclidean_distances(peaks, peaks)
                    euc_distances[np.where(euc_distances == 0.0)] = 10000
                    numb_clusters = 0
                    very_close = []
                    for e in range(euc_distances.shape[0]):
                        if np.min(euc_distances[e, :]) <= 15:
                            very_close.append(np.min(euc_distances[e, :]))
                        else:
                            numb_clusters += 1

                    very_close = list(dict.fromkeys(very_close))
                    numb_clusters = numb_clusters + len(very_close)
            kmeans = KMeans(n_clusters=numb_clusters).fit(temp)

            # arbitrary minimal cluster dimention in pixel: 64
            if self.pixel_size_final < 2.3:
                min_cluster = 64
            # correction minimal cluster dimension if pixel size >= 2.3 nm
            elif self.pixel_size_final < 3.3:
                min_cluster = 79
            elif self.pixel_size_final < 4.3:
                min_cluster = 94
            elif self.pixel_size_final < 5.3:
                min_cluster = 109
            elif self.pixel_size_final < 6.3:
                min_cluster = 124
            else:
                min_cluster = 139

            # check again the distance between peaks (centers of each cluster)
            if numb_clusters > 1:
                euc_distances = euclidean_distances(kmeans.cluster_centers_,
                                                    kmeans.cluster_centers_)
                euc_distances[np.where(euc_distances == 0.0)] = 10000
                potential_numb_vesicles = 0
                very_close = []
                for e in range(euc_distances.shape[0]):
                    if np.min(euc_distances[e, :]) < 15:
                        very_close.append(np.min(euc_distances[e, :]))
                    if np.min(euc_distances[e, :]) >= 15:
                        potential_numb_vesicles += 1
                very_close = list(dict.fromkeys(very_close))
                potential_numb_vesicles = potential_numb_vesicles + len(
                    very_close)

                # if not each cluster is considered a vesicle
                if potential_numb_vesicles < numb_clusters:
                    cluster_size = []
                    cluster_label = []
                    for k in range(numb_clusters):
                        cluster_size.append((kmeans.labels_ == k).sum())
                        cluster_label.append(k)
                    clu = list(zip(cluster_size, cluster_label))
                    clu.sort(reverse=True)
                    clu = clu[:potential_numb_vesicles]

                    # exclude cluster smaller then min_cluster
                    for k in range(len(clu)):
                        if (kmeans.labels_ == clu[k][1]).sum() > min_cluster:
                            x_labels.append(
                                kmeans.cluster_centers_[clu[k][1]][1])
                            y_labels.append(
                                kmeans.cluster_centers_[clu[k][1]][0])

                # if each cluster is cosidered a vesicle
                else:
                    for k in range(len(kmeans.cluster_centers_)):

                        # exclude cluster smaller then min_cluster
                        if (kmeans.labels_ == k).sum() > min_cluster:
                            x_labels.append(kmeans.cluster_centers_[k][1])
                            y_labels.append(kmeans.cluster_centers_[k][0])

            # if there is only one cluster
            else:
                for k in range(len(kmeans.cluster_centers_)):

                    # if each cluster is cosidered a vesicle
                    if (kmeans.labels_ == k).sum() > min_cluster:
                        x_labels.append(kmeans.cluster_centers_[k][1])
                        y_labels.append(kmeans.cluster_centers_[k][0])

        # preparation for feeding second (refinement) classifier
        x_labels_semifinal = []
        y_labels_semifinal = []
        window_size_post = 80

        # put padding on image
        np_img_padded = np.zeros((np_img.shape[0] + 80, np_img.shape[1] + 80))
        np_img_padded[40:np_img.shape[0] + 40,
                      40:np_img.shape[1] + 40] = np_img

        # iterate over the detected vesicles
        for det_ves in range(len(x_labels)):
            snapshot = np_img_padded[int(y_labels[det_ves]):
                                     int(y_labels[det_ves]) + 80,
                                     int(x_labels[det_ves]):
                                         int(x_labels[det_ves]) + 80]
            if (snapshot.shape[0] != window_size_post) or (
                    snapshot.shape[1] != window_size_post):
                continue
            snapshot = snapshot.reshape(1, snapshot.shape[0],
                                        snapshot.shape[1])
            if np.max(snapshot) != np.min(snapshot):
                snapshot = (snapshot - np.min(snapshot)) / (
                    np.max(snapshot) - np.min(snapshot))
            snapshot = (snapshot - 0.5) / 0.5
            snapshot = torch.from_numpy(snapshot)
            snapshot = snapshot.unsqueeze(0)

            # feed image patches into the second (refinement) classifier
            if torch.cuda.is_available():
                output = self.model_post.forward(snapshot.float().cuda())
                valuemax, preds = torch.max(output, 1)
                preds = preds.cpu()

            else:
                output = self.model_post.forward(snapshot.float())
                valuemax, preds = torch.max(output, 1)

            if preds == 1:
                x_labels_semifinal.append(x_labels[det_ves])
                y_labels_semifinal.append(y_labels[det_ves])

        ves_area_analysis = self.ves_area_analysis.get()
        if (ves_area_analysis != 'n') and (ves_area_analysis != 'y'):
            ves_area_analysis = 'n'

        # if the estimation of the vesicles area is not requested by the user
        if ves_area_analysis == 'n':
            x_labels_final = list(
                np.array(x_labels_semifinal) / self.pixel_size_final * 2.27)
            y_labels_final = list(
                np.array(y_labels_semifinal) / self.pixel_size_final * 2.27)
            self.coordinates = list(zip(x_labels_final, y_labels_final))

        # if the estimation of the vesicles area is requested by the user
        else:
            ves_area = np.zeros((np_img.shape[0], np_img.shape[1]))
            x_labels_final = []
            y_labels_final = []
            area = []
            for i in range(len(x_labels_semifinal)):
                minimum = 2550000
                radv = 10
                radh = 10
                ho_shift = 0
                ve_shift = 0
                shift = [-3, -2, -1, 0, 1, 2, 3]
                for o in range(7):
                    for v in range(7):
                        shift_ho = shift[o]
                        shift_ve = shift[v]
                        snap = np_img[
                            int(y_labels_semifinal[i] - 20 + shift_ve):
                                int(y_labels_semifinal[i] + 20 + shift_ve),
                                int(x_labels_semifinal[i] - 20 + shift_ho):
                                    int(x_labels_semifinal[i] + 20 + shift_ho)]
                        if snap.shape != (40, 40):
                            continue

                        # check which ellipse or circle ring (size and shape)
                        # matches at best with the vesicle membrane
                        # st_dev is a penality for inhomogeneity
                        for rv in range(6):
                            for rh in range(6):
                                ring = self.drawing_ellipse_circle(
                                    7 + rv, 7 + rh)
                                if (np.abs(rv - rh) < 5):
                                    matrix = ring * snap
                                    st_dev = np.std(
                                        [np.mean(matrix[:10, :10]),
                                         np.mean(matrix[10:, :10]),
                                         np.mean(matrix[:10, 10:]),
                                         np.mean(matrix[10:, 10:])])
                                    area_ring = np.sum(ring == 1)
                                    membrane_value = np.sum(matrix) / area_ring
                                    value_comp = membrane_value + (
                                        0.03 * st_dev)
                                    if value_comp < minimum:
                                        minimum = membrane_value
                                        radv = 7 + rv
                                        radh = 7 + rh
                                        ho_shift = shift_ho
                                        ve_shift = shift_ve

                # adjust the vesicles coordinates
                cy = int(y_labels_semifinal[i] + ve_shift)
                cx = int(x_labels_semifinal[i] + ho_shift)
                ye, xe = np.ogrid[-radv:radv, -radh:radh]
                index_e = xe**2 / (radh**2) + ye**2 / (radv**2) <= 1

                # fill the area of the vesicles and the final labels
                try:
                    ves_area[cy - radv:cy + radv,
                             cx - radh:cx + radh][index_e] = 1
                except IndexError:
                    pass

                x_labels_final.append(cx / self.pixel_size_final * 2.27)
                y_labels_final.append(cy / self.pixel_size_final * 2.27)
                area.append((radv * radh * np.pi) * (2.27**2))
            self.coordinates = list(zip(x_labels_final, y_labels_final))
            self.area = np.array(area)

            # create the mask with the vesicles areas
            black = 0, 0, 0
            pink = 230, 0, 230
            mask = cv2.resize(ves_area, (shape0, shape1))
            mask = Image.fromarray((mask * 255).astype('uint8'))
            mask = ImageOps.grayscale(mask)
            mask = ImageOps.colorize(mask, black, pink)
            mask = mask.convert('RGBA')
            pixeldata = mask.getdata()
            transparent_list = []
            temp = []
            for n, pixel in enumerate(pixeldata):
                if pixel[0] > 0:
                    temp.append(pixel[0])
            for pixel in pixeldata:
                if pixel[0:3] == (0, 0, 0):
                    transparent_list.append((0, 0, 0, 0))
                else:
                    transparent_list.append((pixel[0], pixel[1], pixel[2], 30))
            mask.putdata(transparent_list)
            self.mask = mask

    def drawing_ellipse_circle(self, radius_ve, radius_ho):
        cx = 20
        cy = 20
        ellipse = np.zeros((40, 40))
        re_maj = radius_ve
        ri_maj = re_maj - 3
        re_min = radius_ho
        ri_min = re_min - 3
        ye, xe = np.ogrid[-re_maj:re_maj, -re_min:re_min]
        index_e = xe**2 / (re_min**2) + ye**2 / (re_maj**2) <= 1
        ellipse[cy - re_maj:cy + re_maj, cx - re_min:cx + re_min][index_e] = 1
        yi, xi = np.ogrid[-ri_maj:ri_maj, -ri_min:ri_min]
        index_i = xi**2 / (ri_min**2) + yi**2 / (ri_maj**2) <= 1
        ellipse[cy - ri_maj:cy + ri_maj, cx - ri_min:cx + ri_min][index_i] = 0
        return ellipse

    def manual_correction1(self):

        # delete old windows or images, if any, and create a new window
        try:
            self.canvas1.delete('all')
        except Exception:
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
        try:
            self.toolbar.pack_forget()
            self.canvas2.get_tk_widget().pack_forget()
            self.canvas2 = None
        except Exception:
            pass

        label1 = tk.Label(
            self.master,
            text=(
                'Enter the name of the excel file where the results are stored'
            ))
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label1)

        self.results_to_use = tk.Entry(self.master)
        self.canvas1.create_window(400, 250, window=self.results_to_use)

        label2 = tk.Label(self.master, text='Select one analysed image')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2)

        button1 = tk.Button(self.master, text='Search file',
                            command=self.manual_correction2,
                            font=('Helvetica', '10'))
        self.canvas1.create_window(400, 350, window=button1)

    def manual_correction2(self):

        self.canvas1.destroy()

        # get the image path and the possible mask path
        self.master.filepath = filedialog.askopenfilename(
            initialdir="/", title="Select file")
        image_path = self.master.filepath
        if image_path[-9:] == '_mask.tif':
            image_path = image_path[:-9] + '.tif'

        splitted = image_path.split('.')
        mask_path = image_path.split('.')[0] + '_mask.tif'

        if len(splitted) > 2:
            mask_path = image_path.split('.')[0]
            for s in range(len(splitted) - 2):
                mask_path = mask_path + '.' + image_path.split('.')[s+1]
            mask_path = mask_path + '_mask.tif'
        self.mask_path = mask_path

        # try to open the image to manually correct
        try:
            img = Image.open(image_path)
            self.img_to_correct = img
        except PIL.UnidentifiedImageError:
            print('The selected file is not identified as an image')
            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()
            label1 = tk.Label(
                self.master,
                text='The selected file is not identified as an image')
            label1.config(font=('helvetica', 12))
            self.canvas1.create_window(400, 250, window=label1)
            button1 = tk.Button(self.master, text='Click here to try again',
                                command=self.manual_correction1,
                                font=('Helvetica', '10'))
            self.canvas1.create_window(400, 350, window=button1)

        # search the sheet where the results of the selected image are stored
        list_path = image_path.split('/')
        self.sheet_name = Path(image_path).stem
        result_dir = ''
        for i in range(len(list_path) - 1):
            result_dir = result_dir + list_path[i] + '/'

        # try to read the result file
        error_free = 1
        if (self.results_to_use.get()[-4:] == 'xlsx') or (
                self.results_to_use.get()[-3:] == 'xls'):
            try:
                excel_filename = self.results_to_use.get()
                self.xls = pd.ExcelFile(result_dir + excel_filename)
            except FileNotFoundError:
                error_free = 0
                print('Result file not found')
                self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                         relief='raised')
                self.canvas1.pack()
                label1 = tk.Label(self.master, text='Result file not found')
                label1.config(font=('helvetica', 12))
                self.canvas1.create_window(400, 250, window=label1)
                button1 = tk.Button(self.master,
                                    text='Click here to try again',
                                    command=self.manual_correction1,
                                    font=('Helvetica', '10'))
                self.canvas1.create_window(400, 350, window=button1)
        else:
            try:
                excel_filename = self.results_to_use.get() + '.xlsx'
                self.xls = pd.ExcelFile(result_dir + excel_filename)
            except FileNotFoundError:
                try:
                    excel_filename = self.results_to_use.get() + '.xls'
                    self.xls = pd.ExcelFile(result_dir + excel_filename)
                except FileNotFoundError:
                    error_free = 0
                    print('Result file not found')
                    self.canvas1 = tk.Canvas(self.master, width=800,
                                             height=600, relief='raised')
                    self.canvas1.pack()
                    label1 = tk.Label(self.master,
                                      text='Result file not found')
                    label1.config(font=('helvetica', 12))
                    self.canvas1.create_window(400, 250, window=label1)
                    button1 = tk.Button(self.master,
                                        text='Click here to try again',
                                        command=self.manual_correction1,
                                        font=('Helvetica', '10'))
                    self.canvas1.create_window(400, 350, window=button1)

        if error_free == 1:
            try:
                df_labels = pd.read_excel(self.xls, self.sheet_name, header=0,
                                          engine='openpyxl')
            except ValueError:
                error_free = 0
                print('Analysis of the selected image not found')
                self.canvas1 = tk.Canvas(self.master, width=800,
                                         height=600,  relief='raised')
                self.canvas1.pack()
                label1 = tk.Label(
                    self.master,
                    text='Analysis of the selected image not found')
                label1.config(font=('helvetica', 12))
                self.canvas1.create_window(400, 250, window=label1)

                button1 = tk.Button(self.master,
                                    text='Click here to try again',
                                    command=self.manual_correction1,
                                    font=('Helvetica', '10'))
                self.canvas1.create_window(400, 350, window=button1)

        if error_free == 1:
            if df_labels.columns[-1] == 'Area (nm²)':
                self.ves_area_analysis = 'y'
            else:
                self.ves_area_analysis = 'n'

            # get the coordinates of the vesicles
            x = []
            y = []
            for idx, row in df_labels.iterrows():
                x.append(row['x_values'])
                y.append(row['y_values'])
            x = np.array(x)
            y = np.array(y)

            # plotting image and labels
            self.fig = Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.imshow(img, cmap='gray')
            self.ax.scatter(x, y, c='blue', s=4)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas2 = FigureCanvasTkAgg(self.fig, master=self.master)
            self.toolbar = tkagg.NavigationToolbar2Tk(self.canvas2,
                                                      self.master)
            self.toolbar.update()
            self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

            # the following lines are necessary for the manual correction
            if len(x) != 1:
                self.original_values = np.array(list(zip(x, y)))
            else:
                self.original_values = np.array([[x, y]])
            self.new_values = []
            self.canvas2.mpl_connect('key_press_event',
                                     self.add_or_remove_points)
            self.canvas2.mpl_connect('key_press_event', self.excel_update1)

    def add_or_remove_points(self, event):

        # add a data point in response to keyboard button 'a'
        if event.key == 'a':
            new_xy_datapoint = [event.xdata, event.ydata]

            # define the current data points
            if len(self.new_values) == 0:
                if len(self.original_values) > 0:
                    xy_datapoints = self.original_values
            else:
                xy_datapoints = self.new_values

            # update new values: current data points + new data point
            if (len(self.original_values) > 0) or (len(self.new_values) > 0):
                self.new_values = np.insert(xy_datapoints, 0, new_xy_datapoint,
                                            axis=0)
            else:
                self.new_values = np.array(new_xy_datapoint).reshape(1, -1)

            # plot the new data point
            self.ax.scatter(event.xdata, event.ydata, c='blue', s=4)
            self.fig.canvas.draw()

        # delete a data point in response to keyboard button 'd',
        # if a data point is close enough to the clicking position
        if event.key == 'd':
            if (len(self.new_values) == 0) and (
                    len(self.original_values) == 0):
                pass
            else:
                xy_click = np.array([event.xdata, event.ydata])

                if len(self.new_values) == 0:
                    xy_datapoints = self.original_values
                else:
                    xy_datapoints = self.new_values

                # look for the vesicle (data point) closest to the click
                closest_point = xy_datapoints[0]
                min_euc = 999999
                idx_to_delete = 0
                for idx in range(len(xy_datapoints)):
                    euc_distance = euclidean_distances(
                        xy_datapoints[idx].reshape(1, -1),
                        xy_click.reshape(1, -1))
                    if euc_distance < min_euc:
                        min_euc = euc_distance
                        closest_point = xy_datapoints[idx]
                        idx_to_delete = idx

                # if the closest data point is close enough, it is deleted
                if min_euc < 12:
                    self.new_values = np.delete(xy_datapoints, idx_to_delete,
                                                axis=0)

                    # plot the the deleted data point in red
                    self.ax.scatter(closest_point[0], closest_point[1],
                                    c='red', s=4)
                    self.fig.canvas.draw()

    def excel_update1(self, event):

        # update the result file in response to the keyboard button 'u'
        # if there is anything to update of course)
        # IMPORTANT: for the update to work do not edit the result file!
        if (len(self.new_values) > 0) and (event.key == 'u'):
            self.toolbar.pack_forget()
            self.canvas2.get_tk_widget().pack_forget()
            self.canvas2 = None

            self.canvas1 = tk.Canvas(self.master, width=800, height=600,
                                     relief='raised')
            self.canvas1.pack()

            label1 = tk.Label(
                self.master,
                text='Please, remind me the pixel size (1 side!) in nanometer')
            label1.config(font=('helvetica', 12))

            self.canvas1.create_window(400, 200, window=label1)

            self.pixel_size = tk.Entry(self.master)
            self.canvas1.create_window(400, 250, window=self.pixel_size)

            button_update = tk.Button(self.master,
                                      text='Click here to update your results',
                                      command=self.excel_update2,
                                      font=('Helvetica', '10'))
            self.canvas1.create_window(400, 350, window=button_update)
        else:
            pass

    def excel_update2(self):

        # delete the old window and create a new one
        self.canvas1.delete('all')
        label1 = tk.Label(self.master, text='Updating...')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label1)
        print('Updating...')

        self.pixel_size_final = float(self.pixel_size.get())

        # get the new x,y coordinates
        x, y = zip(*self.new_values)

        # get the nearest neighbor distances (nnds) between vesicles
        X = np.array([x, y]).T
        euc_distances = euclidean_distances(X, X)
        euc_distances[np.where(euc_distances == 0.0)] = 10000
        min_distances = []
        for i in range(euc_distances.shape[0]):
            temp = min(euc_distances[i, :])
            min_distances.append(temp * float(self.pixel_size.get()))

        x_labels_semifinal = np.zeros((len(x)))
        y_labels_semifinal = np.zeros((len(y)))
        for ii in range(len(x)):
            x_labels_semifinal[ii] = x[ii] * self.pixel_size_final / 2.27
            y_labels_semifinal[ii] = y[ii] * self.pixel_size_final / 2.27

        ves_area_analysis = self.ves_area_analysis
        if (ves_area_analysis != 'n') and (ves_area_analysis != 'y'):
            ves_area_analysis = 'n'

        # if the result file does not have the estimation of the vesicles area
        if ves_area_analysis == 'n':
            x_labels_final = list(x_labels_semifinal)
            y_labels_final = list(y_labels_semifinal)

        # if the result file has the estimation of the vesicles area
        else:
            img_corr = self.img_to_correct.resize(
                (int(self.img_to_correct.size[0] *
                     self.pixel_size_final / 2.27),
                 int(self.img_to_correct.size[1] *
                     self.pixel_size_final / 2.27)))
            np_img_corr = np.array(img_corr)
            ves_area = np.zeros((np_img_corr.shape[0], np_img_corr.shape[1]))
            x_labels_final = []
            y_labels_final = []
            area = []
            for i in range(len(x_labels_semifinal)):
                minimum = 2550000
                ho_shift = 0
                ve_shift = 0
                radv = 10
                radh = 10
                shift = [-3, -2, -1, 0, 1, 2, 3]
                for o in range(7):
                    for v in range(7):
                        shift_ho = shift[o]
                        shift_ve = shift[v]
                        snap = np_img_corr[
                            int(y_labels_semifinal[i] - 20 + shift_ve):
                                int(y_labels_semifinal[i] + 20 + shift_ve),
                                int(x_labels_semifinal[i] - 20 + shift_ho):
                                    int(x_labels_semifinal[i] + 20 + shift_ho)]
                        if snap.shape != (40, 40):
                            continue

                        # check which ellipse or circle ring (size and shape)
                        # matches at best with the vesicle membrane
                        # st_dev is a penality for inhomogeneity
                        for rv in range(6):
                            for rh in range(6):
                                ring = self.drawing_ellipse_circle(
                                    7 + rv, 7 + rh)
                                if (np.abs(rv - rh) < 5):
                                    matrix = ring * snap
                                    st_dev = np.std(
                                        [np.mean(matrix[:10, :10]),
                                         np.mean(matrix[10:, :10]),
                                         np.mean(matrix[:10, 10:]),
                                         np.mean(matrix[10:, 10:])])
                                    area_ring = np.sum(ring == 1)
                                    membrane_value = np.sum(matrix) / area_ring
                                    value_comp = membrane_value + (
                                        0.03 * st_dev)
                                    if value_comp < minimum:
                                        minimum = membrane_value
                                        radv = 7 + rv
                                        radh = 7 + rh
                                        ho_shift = shift_ho
                                        ve_shift = shift_ve

                # adjust the vesicles coordinates
                cy = int(y_labels_semifinal[i] + ve_shift)
                cx = int(x_labels_semifinal[i] + ho_shift)
                ye, xe = np.ogrid[-radv:radv, -radh:radh]
                index_e = xe**2 / (radh**2) + ye**2 / (radv**2) <= 1

                # fill the area of the vesicles and the final labels
                try:
                    ves_area[cy - radv:cy + radv,
                             cx - radh:cx + radh][index_e] = 1
                except IndexError:
                    pass

                x_labels_final.append(cx / self.pixel_size_final * 2.27)
                y_labels_final.append(cy / self.pixel_size_final * 2.27)
                area.append((radv * radh * np.pi) * (2.27**2))

            # create the mask with the updated vesicles areas
            black = 0, 0, 0
            pink = 230, 0, 230
            mask = cv2.resize(ves_area,
                              (self.img_to_correct.size[0],
                               self.img_to_correct.size[1]))
            mask = Image.fromarray((mask * 255).astype('uint8'))
            mask = ImageOps.grayscale(mask)
            mask = ImageOps.colorize(mask, black, pink)
            mask = mask.convert('RGBA')
            pixeldata = mask.getdata()
            transparent_list = []
            temp = []
            for n, pixel in enumerate(pixeldata):
                if pixel[0] > 0:
                    temp.append(pixel[0])
            for pixel in pixeldata:
                if pixel[0:3] == (0, 0, 0):
                    transparent_list.append((0, 0, 0, 0))
                else:
                    transparent_list.append((pixel[0], pixel[1], pixel[2], 30))
            mask.putdata(transparent_list)
            mask.save(self.mask_path)

        # prepare the new values to change the sheet related to processed image
        if len(x) != 1:
            if ves_area_analysis == 'y':
                new_filling = np.array(list(zip(x_labels_final, y_labels_final,
                                                min_distances, area)))
            else:
                new_filling = np.array(list(zip(x_labels_final, y_labels_final,
                                                min_distances)))
        else:
            if ves_area_analysis == 'y':
                new_filling = np.array([[x, y, min_distances, area]])
            else:
                new_filling = np.array([[x, y, min_distances]])

        # re-write the result file
        df = pd.read_excel(self.xls, sheet_name=None, header=0,
                           engine='openpyxl')
        with pd.ExcelWriter(self.xls, engine='xlsxwriter') as writer:
            # iterate over the sheets
            for k in df.keys():
                if k == 'Summary results':
                    df_current_sheet = df[k]
                    print(df_current_sheet['Image name'])
                    df_current_sheet.loc[
                        df_current_sheet['Image name'] ==
                        self.sheet_name, 'Vesicles count'] = len(x)
                    df_current_sheet.to_excel(writer, sheet_name=k,
                                              index=False)

                elif k == self.sheet_name:
                    if ves_area_analysis == 'y':
                        df_current_sheet = pd.DataFrame(columns=[
                            'x_values', 'y_values',
                            'Distance to nearest vesicle (nm)', 'Area (nm²)'])
                    else:
                        df_current_sheet = pd.DataFrame(columns=[
                            'x_values', 'y_values',
                            'Distance to nearest vesicle (nm)'])

                    for i, e in enumerate(new_filling):
                        df_current_sheet.loc[str(i)] = e
                    df_current_sheet.to_excel(writer, sheet_name=k,
                                              index=False)
                else:
                    df_current_sheet = df[k]
                    df_current_sheet.to_excel(writer, sheet_name=k,
                                              index=False)

        # save and close
        writer.save()
        writer.close()

        self.canvas1.delete('all')
        label2 = tk.Label(self.master, text='Update done!')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2)
        print('Update done!')
