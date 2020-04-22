# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:35:25 2020

@author: imbroscb
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
import seaborn as sns
from tkinter import filedialog
from vesicle_classifier import MultiClass
import xlsxwriter
import pandas as pd
from time import sleep
import os
from torchvision import transforms
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import scipy
import cv2
from scipy import ndimage
from PIL import Image, ImageOps



#%%

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

        analysis = tk.Menu(menu)
        display = tk.Menu(menu)

        analysis.add_command(label="Vesicles detection", command=self.start_analysis_1)
        analysis.add_command(label='Exit',command=self.exit_program)        
        display.add_command(label='Display detection on image',command=self.image_plus_results1)
        display.add_command(label='Display graphic results',command=self.graphic_results)
   
        menu.add_cascade(label="Analysis", menu=analysis)
        menu.add_cascade(label="Display", menu=display)
     
    def start_analysis_1(self):

        try: 
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget() 
        except:
            pass

        self.canvas1 = tk.Canvas(self.master, width = 800, height = 600,  relief = 'raised')
        self.canvas1.pack()

        label1 = tk.Label(self.master, text='Enter experiment name')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 100, window=label1)
        
        self.experiment_name=tk.Entry (self.master) 
        self.canvas1.create_window(400, 150, window=self.experiment_name)
        
        label2 = tk.Label(self.master, text='Enter the pixel size (1 side!) in nanometer')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label2)
        
        self.pixel_size = tk.Entry (self.master) 
        self.canvas1.create_window(400, 250, window=self.pixel_size)
        
        label4 = tk.Label(self.master, text='Select the directory where images are located')
        label4.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 350, window=label4) 
        
        button1=tk.Button(self.master,text='Search directory',command=self.start_analysis_2,font=('Helvetica', '10'))
        self.canvas1.create_window(400,400,window=button1)
                        
    def start_analysis_2(self):

        self.canvas1.delete('all')     
        self.master.directory =filedialog.askdirectory()
        sleep(1)
        
        self.var = tk.StringVar()
        label1 = tk.Label(self.master, textvariable=self.var)
        label1.config(font=('helvetica', 14))
        self.canvas1.create_window(400, 300, window=label1)        
        self.var.set('Analysis is starting...')  
        self.master.update_idletasks()
        sleep(1)

        # load model           
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        PATH=self.master.directory+'/'+'model.pth'
        
        if torch.cuda.is_available():         
            self.model=MultiClass(out=2).to(device)
            self.model.load_state_dict(torch.load(PATH))
        else:
            self.model=MultiClass(out=2)
            self.model.load_state_dict(torch.load(PATH,map_location=device))    
            
        self.model.eval()        
                
        image_dir=self.master.directory                       
        excel_name=self.experiment_name.get()+'.xlsx'
        
        book = xlsxwriter.Workbook(image_dir+'/'+excel_name)
        main_result_sheet=book.add_worksheet('Summary results')
        main_result_sheet.write(0,0,'Image name') 
        main_result_sheet.write(0,1,'Vesicles count') 
        
        self.pixel_size_final=float(self.pixel_size.get())
        
        # start with analysis           
        self.counter=1              
        for file in os.listdir(image_dir):

            image_name=file.split('.')[0]

            if (file.split('.')[-1]=='xlsx') or (file.split('.')[-1]=='xls'):
                continue    
            if ('_mask' in file.split('.')[0]) or (file.split('.')[-1]=='pth'):
                continue
            
            try:
                self.img_to_analyse=Image.open(image_dir+'/'+file) 
            except:
                continue

            self.var.set('Processing image number: ' + str(self.counter))            
            self.master.update_idletasks()

            sheet = book.add_worksheet(image_name)
            sleep(1)         
            
            # getting the x,y coordinates of the detected vesicles
            self.sliding_detection()
            self.mask.save(image_dir+'/'+image_name+'_mask.tif')
            
            # getting the min distances between vesicles
            x,y=zip(*self.coordinates)
            X=np.array([x,y]).T
            euc_distances=euclidean_distances(X, X)

            euc_distances[np.where(euc_distances==0.0)]=10000
            min_distances=[]
            for i in range(euc_distances.shape[0]):
                temp=min(euc_distances[i,:])
                min_distances.append(temp*float(self.pixel_size.get()))
    
            # extract the x,y coordinates and the distances of the detected vesicles
            for i,e in enumerate(self.coordinates):
                
                sheet.write(i+1,0,e[0])
                sheet.write(i+1,1,e[1])
                sheet.write(i+1,2,min_distances[i])

            sheet.write(0,0,'x_values')
            sheet.write(0,1,'y_values')            
            sheet.write(0,2,'Distance to nearest vesicle (nm)')

            # add summary results
            main_result_sheet.write(self.counter,0,image_name)
            main_result_sheet.write(self.counter,1,len(self.coordinates))
            
            self.counter +=1
                
        self.var.set('Done!')
        print('Done!')
        self.master.update_idletasks()  
    
        book.close() 
        
    def exit_program(self):
        self.master.quit()  
        self.master.destroy()

    
    def image_plus_results1(self):
        
        try: 
            self.canvas1.delete('all')
        except:
            self.canvas1 = tk.Canvas(self.master, width = 800, height = 600,  relief = 'raised')
            self.canvas1.pack()

        try: 
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget() 
        except:
            pass
        
        label1 = tk.Label(self.master, text='Enter the name of the excel file where results are stored')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label1)
        
        self.results_to_use = tk.Entry (self.master) 
        self.canvas1.create_window(400, 250, window=self.results_to_use)
        
        label2 = tk.Label(self.master, text='Select one analysed image')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2) 
        
        button1=tk.Button(self.master,text='Search file',command=self.image_plus_results2,font=('Helvetica', '10'))
        self.canvas1.create_window(400,350,window=button1)
                       
    def image_plus_results2(self):
        
        self.canvas1.destroy()
        
        #getting the image path
        self.master.filepath=filedialog.askopenfilename(initialdir = "/",title = "Select file")
        image_path=self.master.filepath
        if image_path[-9:]=='_mask.tif':
            image_path=image_path[:-9]+'.tif'
            
        mask_path=image_path.split('.')[0]+'_mask.tif'
        img=Image.open(image_path)
        mask=Image.open(mask_path)
                
        list_path=image_path.split('/')
        result_dir=''
        for i in range(len(list_path)-1):
            temp=list_path[i]+'/'
            result_dir=result_dir+temp
               
        sheet_name=list_path[-1]
        sheet_name=sheet_name.split('.')[0]
        
        if (self.results_to_use.get()[-4:]=='xlsx') or (self.results_to_use.get()[-4:]=='xls'):
            excel_filename=self.results_to_use.get()
        else:    
            excel_filename=self.results_to_use.get() +'.xlsx'

        xls = pd.ExcelFile(result_dir+excel_filename)
        df_labels = pd.read_excel(xls,sheet_name,header=0)
        x=[]
        y=[]
        for idx,row in df_labels.iterrows():
            x.append(row['x_values'])
            y.append(row['y_values'])
            
        x=np.array(x)
        y=np.array(y)

        # plotting image and labels
        fig=Figure(figsize=(10,8))
        image=fig.add_subplot(1,1,1)
        image.imshow(img,cmap='gray') 
        image.imshow(mask)
        image.scatter(x,y,c='white',s=4)
    
        try:
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()           
            self.canvas3=FigureCanvasTkAgg(fig,master=self.master)        
            toolbar=tkagg.NavigationToolbar2Tk(self.canvas3, self.master)
            self.toolbar.update()
            self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
            
        except AttributeError:             
            self.canvas3=FigureCanvasTkAgg(fig,master=self.master)
            self.toolbar=tkagg.NavigationToolbar2Tk(self.canvas3, self.master)
            self.toolbar.update()
            self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
    
    def graphic_results(self):
        
        try:
            self.canvas1.delete('all')
        except:
            self.canvas1 = tk.Canvas(self.master, width = 800, height = 600,  relief = 'raised')
            self.canvas1.pack()
        try:
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()  
          
        except AttributeError:                      
            pass
        
        label1 = tk.Label(self.master, text='Enter the name of the excel file where results are stored')
        label1.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label1)
        
        self.results_to_use = tk.Entry (self.master) 
        self.canvas1.create_window(400, 250, window=self.results_to_use)
        
        label2 = tk.Label(self.master, text='Select the directory where all the results files are located')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2) 
        
        button1=tk.Button(self.master,text='Search',command=self.plot,font=('Helvetica', '10'))
        self.canvas1.create_window(400,350,window=button1)
        
    def plot(self):
        self.canvas1.destroy()
        self.master.result_dir =filedialog.askdirectory()
        result_dir=self.master.result_dir
        
        ves_count=[]
        result_filename=[]
        average_min_distance=[]
        
        if (self.results_to_use.get()[-4:]=='xlsx') or (self.results_to_use.get()[-4:]=='xls'):
            excel_filename=self.results_to_use.get()
        else:    
            excel_filename=self.results_to_use.get() +'.xlsx'

        xls = pd.ExcelFile(result_dir+'/'+excel_filename)

        temp = pd.read_excel(xls,'Summary results',header=0)
        temp = temp['Vesicles count']
        ves_count.append(np.array(temp))
        result_filename.append(excel_filename.split('.')[0])

        sheet_list=xls.sheet_names
        av_dist_per_exp=np.zeros((len(sheet_list)-1))
        for i in range(len(sheet_list)-1):
            temp = pd.read_excel(xls,sheet_list[i+1],header=0)
            temp = temp['Distance to nearest vesicle (nm)']
            av_dist_per_exp[i]=np.mean(np.array(temp))
        average_min_distance.append(av_dist_per_exp)
       
        tuple1_list=[]
        tuple2_list=[]
        i=0
        for f in result_filename:
            exp_name=f
            for j in range(len(ves_count[i])):
                tuple1_list.append((exp_name,ves_count[i][j]))
                tuple2_list.append((exp_name,average_min_distance[i][j]))
            i+=1
        df_data1=pd.DataFrame(tuple1_list,columns=['Experiment','Vesicles count'])
        df_data2=pd.DataFrame(tuple2_list,columns=['Experiment','Nearest neighbor distance (nm)'])

        fig=Figure(figsize=(10,20))
        ax1=fig.add_subplot(1,2,1)
        a=sns.swarmplot(x='Experiment',y='Vesicles count',data=df_data1,ax=ax1) 
        a.set_xlabel('Name of experiment',fontsize=18)
        a.set_ylabel('Vesicles count',fontsize=18)
        a.tick_params(labelsize=18)
        ax2=fig.add_subplot(1,2,2)
        b=sns.swarmplot(x='Experiment',y='Nearest neighbor distance (nm)',data=df_data2,ax=ax2)        
        b.set_xlabel('Name of experiment',fontsize=18)
        b.set_ylabel('Mean nearest neighbor (nm)',fontsize=18)
        b.tick_params(labelsize=18)
        
        self.canvas3=FigureCanvasTkAgg(fig,master=self.master)        
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
                
    def sliding_detection(self):
        
        shape0=self.img_to_analyse.size[0]
        shape1=self.img_to_analyse.size[1]

        img=self.img_to_analyse.resize((int(self.img_to_analyse.size[0]*self.pixel_size_final/2.277),int(self.img_to_analyse.size[1]*self.pixel_size_final/2.277)))
        np_img=np.array(img) 
        if len(np_img.shape)>2:
            np_img=cv2.cvtColor(np_img,cv2.COLOR_BGR2GRAY)
            
        sliding_size=4 
        window_size=40
        x_coord=[]
        y_coord=[]
        
        # create a probability map
        p_map=np.zeros((int(np_img.shape[0]/sliding_size),int(np_img.shape[1]/sliding_size)))
        print('Processing image number: {:.0f}'.format(self.counter))
        for x in range(0,np_img.shape[1],sliding_size):

            percentage=x/int(np_img.shape[1])*100
            self.var.set('Processing image number: ' + str(self.counter) + ', done: {:.1f} %'.format(percentage))
            
            print('Done: {:.1f} %'.format(percentage))
            for y in range(0,np_img.shape[0],sliding_size):
                snapshot=np_img[y:y+window_size,x:x+window_size]
                
                if (snapshot.shape[0] !=window_size) or (snapshot.shape[1] != window_size):
                    continue

                snapshot=snapshot.reshape(1,snapshot.shape[0],snapshot.shape[1])
                if np.max(snapshot) != np.min(snapshot):
                    snapshot=(snapshot-np.min(snapshot))/(np.max(snapshot)-np.min(snapshot))
                snapshot=(snapshot-0.5)/0.5
                snapshot=torch.from_numpy(snapshot)
                snapshot=snapshot.unsqueeze(0)
    
                if torch.cuda.is_available():
                    output=self.model.forward(snapshot.float().cuda())
                    valuemax,preds=torch.max(output,1)
                    valuemin,_=torch.min(output,1)
                    valuemax=valuemax.cpu()
                    valuemin=valuemin.cpu()
                    preds=preds.cpu()  
                
                else:
                    output=self.model.forward(snapshot.float())
                    valuemax,preds=torch.max(output,1)
                    valuemin,_=torch.min(output,1)
 
                if preds==1:
                    valuemax=valuemax.data.numpy()
                    valuemin=valuemin.data.numpy()
                    pvalue=np.exp(valuemax)/(np.exp(valuemax)+np.exp(valuemin))
                    p_map[int((y+20)/sliding_size),int((x+20)/sliding_size)]=pvalue

        # resize pmap and create a mask
        proc_pmap = cv2.resize(p_map,(np_img.shape[1],np_img.shape[0]))
        proc_pmap=cv2.blur(proc_pmap,(3,3))
        proc_pmap=ndimage.gaussian_filter(proc_pmap,0.05)
        proc_pmap=(proc_pmap / (np.max(proc_pmap))) *255
        
        for xx in range(proc_pmap.shape[0]):
            for yy in range(proc_pmap.shape[1]):
                if proc_pmap[xx,yy]<255/100*20: 
                    proc_pmap[xx,yy]=0

        black=0,0,0
        pink=230,0,230

        mask = cv2.resize(proc_pmap,(shape0,shape1))     
        mask=Image.fromarray(mask)
        mask=ImageOps.grayscale(mask)
        mask=ImageOps.colorize(mask,black,pink)       
        mask=mask.convert('RGBA')
        
        pixeldata = mask.getdata()
        transparent_list=[]
                    
        temp=[]
        for n,pixel in enumerate(pixeldata):
            if pixel[0]>0:
                temp.append(pixel[0])
        
        media=np.mean(temp)
        stdev=np.std(temp)
            
        for pixel in pixeldata:
            if pixel[0:3]==(0,0,0):
                transparent_list.append((0,0,0,0))
            elif pixel[0]<media-stdev:
                transparent_list.append((pixel[0],pixel[1],pixel[2],30))
            elif pixel[0]<media:
                transparent_list.append((pixel[0],pixel[1],pixel[2],50))
            elif pixel[0]<media+stdev:
                transparent_list.append((pixel[0],pixel[1],pixel[2],70))
            else:
                transparent_list.append((pixel[0],pixel[1],pixel[2],90))
                    
        mask.putdata(transparent_list)
               
        # Counting the vesicles
        labelarray,counts=ndimage.measurements.label(proc_pmap)

        x_labels=[]
        y_labels=[]
        
        for i in range(counts):
        
            x,y=np.where(labelarray==i+1)        
            temp=[]       
            if type(x) == int:
                temp.append(x,y)        
            else:
                for j in range(len(x)):
                    temp.append((x[j],y[j]))
                    
            euc_distances=euclidean_distances(temp,temp)  
            max_distance=np.max(euc_distances)
                    
            potential_peaks=1
            
            # assume if max_distance>25 that there are at least 2 vesicles in the label
            if max_distance >25:
            
                # detect if there are more than 2 peaks (assuming each peak is 1 vesicle)        
                peaks=[]
                for j in range(len(x)):
        
                    temp_peak=proc_pmap[x[j],y[j]]
                    temp_peak_idx=(x[j],y[j])
                    challenge_peak=np.zeros((8))
                    gapx=np.array([-1,-1,-1,0,0,1,1,1])
                    gapy=np.array([-1,0,1,-1,1,-1,0,1])
                    
                    # look aroung temp_peak if it is a peak indeed
                    for g in range(8):
                        challenge_peak[g]=proc_pmap[x[j]+gapx[g],y[j]+gapy[g]]
                    # if no pixel value aroung temp_peak is bigger than temp_peak is a peak!
                    if (np.max(challenge_peak)<=temp_peak):
                        peaks.append(temp_peak_idx)
                    
                # calculate the distance between the peaks 
                if len(peaks)>1:      
   
                    euc_distances=euclidean_distances(peaks,peaks)  
                    euc_distances[np.where(euc_distances==0.0)]=10000
                    potential_peaks=0
                    minimi=[]
                    for e in range(euc_distances.shape[0]):

                        if np.min(euc_distances[e,:])<=15:
                            minimi.append(np.min(euc_distances[e,:]))                              
                        else:
                            potential_peaks+=1
                    
                    minimi=list(dict.fromkeys(minimi))
                    potential_peaks=potential_peaks+len(minimi)            

            kmeans=KMeans(n_clusters=potential_peaks).fit(temp)
        
            # check again the distance between peaks (centers of each cluster)
            if potential_peaks>1:      
                euc_distances=euclidean_distances(kmeans.cluster_centers_,kmeans.cluster_centers_)  
                euc_distances[np.where(euc_distances==0.0)]=10000
                potential_peaks_2=0
                minimi=[]
                for e in range(euc_distances.shape[0]):
                    if np.min(euc_distances[e,:])<15:
                        minimi.append(np.min(euc_distances[e,:]))   
                    if np.min(euc_distances[e,:])>=15:
                        potential_peaks_2 +=1
                minimi=list(dict.fromkeys(minimi))
                potential_peaks_2=potential_peaks_2+len(minimi)
                
                # not each cluster is considered a vesicle
                if potential_peaks_2< potential_peaks:
                    cluster_size=[]
                    cluster_label=[]
                    for k in range(potential_peaks):
                        cluster_size.append((kmeans.labels_==k).sum())
                        cluster_label.append(k)
                    clu=list(zip(cluster_size,cluster_label))
                    clu.sort(reverse=True) 
                    clu=clu[:potential_peaks_2] 
                    
                    for k in range(len(clu)):
                        if (kmeans.labels_==clu[k][1]).sum() >64: 
                            x_labels.append(kmeans.cluster_centers_[clu[k][1]][1]/self.pixel_size_final*2.277)
                            y_labels.append(kmeans.cluster_centers_[clu[k][1]][0]/self.pixel_size_final*2.277)
                
                # if each cluster is cosidered a  vesicle
                else:
                    
                    for k in range(len(kmeans.cluster_centers_)):
                        if (kmeans.labels_==k).sum() >64: 
                            x_labels.append(kmeans.cluster_centers_[k][1]/self.pixel_size_final*2.277)
                            y_labels.append(kmeans.cluster_centers_[k][0]/self.pixel_size_final*2.277)
            
            # if there is only 1 cluster
            else:
                
                for k in range(len(kmeans.cluster_centers_)):
                    if (kmeans.labels_==k).sum() >64: 
                        x_labels.append(kmeans.cluster_centers_[k][1]/self.pixel_size_final*2.277)
                        y_labels.append(kmeans.cluster_centers_[k][0]/self.pixel_size_final*2.277)
           
        coordinates=list(zip(x_labels,y_labels)) 
        
        self.coordinates=coordinates
        self.mask=mask
    
