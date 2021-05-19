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
from vesicle_classifier import MultiClass, MultiClassPost
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

        run_analysis = tk.Menu(menu)
        post_analysis = tk.Menu(menu)
        display = tk.Menu(menu)
        
        menu.add_cascade(label="Analysis", menu=run_analysis)
        menu.add_cascade(label="Results_check", menu=display)  

        
        run_analysis.add_command(label="Vesicles detection", command=self.start_analysis_1)
        run_analysis.add_separator()
        run_analysis.add_command(label='Exit',command=self.exit_program)   

        
        display.add_command(label='Display detection on image',command=self.image_plus_results1)
        display.add_command(label='Display graphic results',command=self.graphic_results)
        display.add_command(label='Manual correction',command=self.manual_correction1)

     
    def start_analysis_1(self):

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
        PATH_post=self.master.directory+'/'+'model_post.pth'

        if torch.cuda.is_available():         
            self.model=MultiClass(out=2).to(device)
            self.model.load_state_dict(torch.load(PATH))
            self.model_post=MultiClassPost(out=2).to(device)
            self.model_post.load_state_dict(torch.load(PATH_post))            
        else:
            self.model=MultiClass(out=2)
            self.model.load_state_dict(torch.load(PATH,map_location=device))    
            self.model_post=MultiClassPost(out=2)
            self.model_post.load_state_dict(torch.load(PATH_post,map_location=device))  
            
        self.model.eval()  
        self.model_post.eval()
                
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

            splitted=file.split('.')
            image_name=file.split('.')[0]
            
            if len(splitted)>2:
                for s in range(len(splitted)-2):
                    image_name=image_name+'.'+ file.split('.')[s+1]

            if (file.split('.')[-1]=='xlsx') or (file.split('.')[-1]=='xls'):
                continue    
            if len(file.split('.'))==1:
                continue                
            elif ('_mask' in file.split('.')[-2]) or (file.split('.')[-1]=='pth'):
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
            if len(self.coordinates)>0:
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
                    sheet.write(i+1,3,self.area[i])

            sheet.write(0,0,'x_values')
            sheet.write(0,1,'y_values')            
            sheet.write(0,2,'Distance to nearest vesicle (nm)')
            sheet.write(0,3,'Area (nm²)')


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
    
        splitted=image_path.split('.')
        mask_path=image_path.split('.')[0]+'_mask.tif'
        
        if len(splitted)>2:
            mask_path=image_path.split('.')[0]
            for s in range(len(splitted)-2):
                mask_path=mask_path+'.'+image_path.split('.')[s+1]
            mask_path=mask_path+'_mask.tif'
        
        img=Image.open(image_path)
        mask=Image.open(mask_path)
                
        list_path=image_path.split('/')
        result_dir=''
        for i in range(len(list_path)-1):
            temp=list_path[i]+'/'
            result_dir=result_dir+temp
               
        sheet_name=list_path[-1]
        splitted=sheet_name.split('.')
        self.sheet_name=sheet_name.split('.')[0]
        
        if len(splitted)>2:
            for s in range(len(splitted)-2):
                self.sheet_name=self.sheet_name+'.'+sheet_name.split('.')[s+1]
        
        if (self.results_to_use.get()[-4:]=='xlsx') or (self.results_to_use.get()[-3:]=='xls'):
            excel_filename=self.results_to_use.get()
        else:    
            excel_filename=self.results_to_use.get() +'.xlsx'

        self.xls = pd.ExcelFile(result_dir+excel_filename)
        df_labels = pd.read_excel(self.xls,self.sheet_name,header=0)
        x=[]
        y=[]
        for idx,row in df_labels.iterrows():
            x.append(row['x_values'])
            y.append(row['y_values'])
            
        x=np.array(x)
        y=np.array(y)
        
        # plotting image and labels
        self.fig=Figure(figsize=(10,8))
        self.ax=self.fig.add_subplot(1,1,1)
        self.ax.imshow(img,cmap='gray') 
        self.ax.imshow(mask)
        self.ax.scatter(x,y,c='white',s=4)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
        try:
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()           
            self.canvas3=FigureCanvasTkAgg(self.fig,master=self.master)        
            toolbar=tkagg.NavigationToolbar2Tk(self.canvas3, self.master)
            self.toolbar.update()
            self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
            
        except AttributeError:             
            self.canvas3=FigureCanvasTkAgg(self.fig,master=self.master)
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
        
        #label1 = tk.Label(self.master, text='Enter the name of the excel file where results are stored')
        #label1.config(font=('helvetica', 12))
        #self.canvas1.create_window(400, 200, window=label1)
        
        #self.results_to_use = tk.Entry (self.master) 
        #self.canvas1.create_window(400, 250, window=self.results_to_use)
        
        label2 = tk.Label(self.master, text='Select the directory where all the results files are located')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 200, window=label2) 
        
        button1=tk.Button(self.master,text='Search',command=self.plot,font=('Helvetica', '10'))
        self.canvas1.create_window(400,300,window=button1)
        

            
    def plot(self):
        self.canvas1.destroy()
        self.master.result_dir =filedialog.askdirectory()
        result_dir=self.master.result_dir
        
        experiments_list=[]
        for file in os.listdir(result_dir):
            if file[-4:]=='xlsx': 
                experiments_list.append(file)
            elif file[-3:]=='xls':
                experiments_list.append(file)
            else:
                continue
            
        ves_count=[]
        result_filename=[]
        average_min_distance=[]
        average_area=[]
        
        for experiment_name in experiments_list:
            
            xls = pd.ExcelFile(result_dir+'/'+experiment_name)
            temp = pd.read_excel(xls,'Summary results',header=0)
            temp = temp['Vesicles count']
            ves_count.append(np.array(temp))
            
            if experiment_name[-4:]=='xlsx':                
                result_filename.append(experiment_name[:-5])
            elif experiment_name[-3:]=='xls':                
                result_filename.append(experiment_name[:-4])

            sheet_list=xls.sheet_names
            av_dist_per_exp=np.zeros((len(sheet_list)-1))
            av_area_per_exp=np.zeros((len(sheet_list)-1))
            
            for i in range(len(sheet_list)-1):           
                
                temp = pd.read_excel(xls,sheet_list[i+1],header=0)
                temp_nn = temp['Distance to nearest vesicle (nm)']
                temp_area=temp['Area (nm²)']
                av_dist_per_exp[i]=np.mean(np.array(temp_nn))
                av_area_per_exp[i]=np.mean(np.array(temp_area))
        
            average_min_distance.append(av_dist_per_exp)
            average_area.append(av_area_per_exp)
       
        tuple1_list=[]
        tuple2_list=[]
        tuple3_list=[]
        i=0
        for f in result_filename:

            exp_name=f
            for j in range(len(ves_count[i])):
                tuple1_list.append((exp_name,ves_count[i][j]))
                tuple2_list.append((exp_name,average_min_distance[i][j]))
                tuple3_list.append((exp_name,average_area[i][j]))                                
            i+=1
        
        df_data1=pd.DataFrame(tuple1_list,columns=['Experiment','Vesicles count'])
        df_data2=pd.DataFrame(tuple2_list,columns=['Experiment','Nearest neighbor distance (nm)'])
        df_data3=pd.DataFrame(tuple3_list,columns=['Experiment','Area (nm²)'])

        fig=Figure(figsize=(10,20))
        ax1=fig.add_subplot(1,3,1)
        a=sns.swarmplot(x='Experiment',y='Vesicles count',data=df_data1,ax=ax1) 
        a.set_xlabel('Name of experiment',fontsize=12,position=(9,0.3))
        a.set_ylabel('Vesicles count',fontsize=12)
        a.tick_params(labelsize=12)
        ax2=fig.add_subplot(1,3,2)
        b=sns.swarmplot(x='Experiment',y='Nearest neighbor distance (nm)',data=df_data2,ax=ax2)        
        b.set_xlabel('Name of experiment',fontsize=12,position=(9,3.6))
        b.set_ylabel('Mean nearest neighbor (nm)',fontsize=12)
        b.tick_params(labelsize=12)
        ax3=fig.add_subplot(1,3,3)
        c=sns.swarmplot(x='Experiment',y='Area (nm²)',data=df_data3,ax=ax3)        
        c.set_xlabel('Name of experiment',fontsize=12,position=(9,6.9))
        c.set_ylabel('Mean area (nm²)',fontsize=12)
        c.tick_params(labelsize=12)
        
        self.canvas3=FigureCanvasTkAgg(fig,master=self.master)        
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
                
    def sliding_detection(self):
        
        shape0=self.img_to_analyse.size[0]
        shape1=self.img_to_analyse.size[1]

        img=self.img_to_analyse.resize((int(self.img_to_analyse.size[0]*self.pixel_size_final/2.27),int(self.img_to_analyse.size[1]*self.pixel_size_final/2.27)))
        np_img=np.array(img) 
        if len(np_img.shape)>2:
            np_img=cv2.cvtColor(np_img,cv2.COLOR_BGR2GRAY)
            
        sliding_size=4 
        window_size=40
        x_coord=[]
        y_coord=[]
        
        np_img_padded=np.zeros((np_img.shape[0]+40,np_img.shape[1]+40))
        np_img_padded[20:np_img.shape[0]+20,20:np_img.shape[1]+20]=np_img
         
    
        # create a probability map
        p_map=np.zeros((int(np_img.shape[0]/sliding_size),int(np_img.shape[1]/sliding_size)))
        print('Processing image number: {:.0f}'.format(self.counter))
        for x in range(0,np_img.shape[1],sliding_size):

            percentage=x/int(np_img.shape[1])*100
            self.var.set('Processing image number: ' + str(self.counter) + ', done: {:.1f} %'.format(percentage))
            
            print('Done: {:.1f} %'.format(percentage))
            for y in range(0,np_img.shape[0],sliding_size):
                snapshot=np_img_padded[y+20:y+window_size+20,x+20:x+window_size+20]
                
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
        proc_pmap = cv2.resize(p_map,(np_img.shape[1],np_img.shape[0]))#,interpolation=cv2.INTER_NEAREST)
        proc_pmap=cv2.blur(proc_pmap,(3,3))
        
        if np.max(proc_pmap)>0:
            proc_pmap=(proc_pmap / (np.max(proc_pmap))) *255
        
        for xx in range(proc_pmap.shape[0]):
            for yy in range(proc_pmap.shape[1]):
                if proc_pmap[xx,yy]<255/100*20: 
                    proc_pmap[xx,yy]=0                


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
            
            try:
                euc_distances=euclidean_distances(temp,temp)  
            except MemoryError:
                break
            
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
            
            #arbitrary min cluster dimention in pixel (64)
            if self.pixel_size_final<2.3:                
                min_cluster=64
            # correction min dimention cluster for images with rel. low resolution
            elif self.pixel_size_final<3.3: 
                min_cluster=79
            elif self.pixel_size_final<4.3:
                min_cluster=94
            elif self.pixel_size_final<5.3:
                min_cluster=109
            elif self.pixel_size_final<6.3:
                min_cluster=124
               
            else:
                min_cluster=139 
            
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
                        if (kmeans.labels_==clu[k][1]).sum() >min_cluster:
                            x_labels.append(kmeans.cluster_centers_[clu[k][1]][1])
                            y_labels.append(kmeans.cluster_centers_[clu[k][1]][0])
             
                # if each cluster is cosidered a  vesicle
                else:
                    
                    for k in range(len(kmeans.cluster_centers_)):
                        if (kmeans.labels_==k).sum() >min_cluster:
                            x_labels.append(kmeans.cluster_centers_[k][1])
                            y_labels.append(kmeans.cluster_centers_[k][0])
            
            # if there is only 1 cluster
            else:
                
                for k in range(len(kmeans.cluster_centers_)):
                    if (kmeans.labels_==k).sum() >min_cluster:  
                        x_labels.append(kmeans.cluster_centers_[k][1])
                        y_labels.append(kmeans.cluster_centers_[k][0])                   
        

        x_labels_semifinal=[]
        y_labels_semifinal=[]
        
        window_size_post=80
        np_img_padded=np.zeros((np_img.shape[0]+80,np_img.shape[1]+80))
        np_img_padded[40:np_img.shape[0]+40,40:np_img.shape[1]+40]=np_img

        for det_ves in range(len(x_labels)):

            snapshot=np_img_padded[int(y_labels[det_ves]):int(y_labels[det_ves])+80,int(x_labels[det_ves]):int(x_labels[det_ves])+80]
        
            if (snapshot.shape[0] !=window_size_post) or (snapshot.shape[1] != window_size_post):
                continue

            snapshot=snapshot.reshape(1,snapshot.shape[0],snapshot.shape[1])
            if np.max(snapshot) != np.min(snapshot):
                snapshot=(snapshot-np.min(snapshot))/(np.max(snapshot)-np.min(snapshot))
        
            snapshot=(snapshot-0.5)/0.5
            snapshot=torch.from_numpy(snapshot)
            snapshot=snapshot.unsqueeze(0)

            if torch.cuda.is_available():
                output=self.model_post.forward(snapshot.float().cuda())
                valuemax,preds=torch.max(output,1)
                valuemin,_=torch.min(output,1)
                preds=preds.cpu()  
                valuemax=valuemax.cpu()
                valuemin=valuemin.cpu()
      
            else:
                output=self.model_post.forward(snapshot.float())
                valuemax,preds=torch.max(output,1)
                valuemin,_=torch.min(output,1)
           
            valuemax=valuemax.data.numpy()
            valuemin=valuemin.data.numpy()                 
            
            if preds==1:
                pvalue=np.exp(valuemax)/(np.exp(valuemax)+np.exp(valuemin))
            
            else:
                pvalue=np.exp(valuemin)/(np.exp(valuemax)+np.exp(valuemin))                    
                
            if preds==1:
            #if pvalue>=0.5:
                x_labels_semifinal.append(x_labels[det_ves])
                y_labels_semifinal.append(y_labels[det_ves])
        
        # here calculate area and make mask
        ves_area=np.zeros((np_img.shape[0],np_img.shape[1]))        

        x_labels_final=[]
        y_labels_final=[]
        area=[]

        for i in range(len(x_labels_semifinal)):            
            minimo=2550000
            radv=10
            radh=10
            ho_shift=0
            ve_shift=0
            shift=[-3,-2,-1,0,1,2,3]
            for o in range(7):
                for v in range(7):
                    shift_ho=shift[o]
                    shift_ve=shift[v]
                    snap=np_img[int(y_labels_semifinal[i]-20+shift_ve):int(y_labels_semifinal[i]+20+shift_ve),int(x_labels_semifinal[i]-20+shift_ho):int(x_labels_semifinal[i]+20+shift_ho)]
                    
                    if snap.shape != (40,40):
                        continue
                    
                    for rv in range(6):
                        for rh in range(6):
                            ring=self.drawing_ellipse(7+rv,7+rh)
                            if (np.abs(rv-rh)<5):
                                matrice=ring*snap
                                st_dev=np.std([np.mean(matrice[:10,:10]),np.mean(matrice[10:,:10]),np.mean(matrice[:10,10:]),np.mean(matrice[10:,10:])])
                                area_ring=np.sum(ring==1)
                                membrane_value=np.sum(matrice)/area_ring
                                value_comp=membrane_value+(0.03*st_dev)
                                if value_comp<minimo:
                                    minimo=membrane_value
                                    radv=7+rv
                                    radh=7+rh
                                    ho_shift=shift_ho
                                    ve_shift=shift_ve
                
            cy=int(y_labels_semifinal[i]+ve_shift) 
            cx=int(x_labels_semifinal[i]+ho_shift)
            ye,xe=np.ogrid[-radv:radv,-radh:radh]
            index_e=xe**2/(radh**2)+ye**2/(radv**2) <=1
            
            try:
                ves_area[cy-radv:cy+radv,cx-radh:cx+radh][index_e]=1
            except IndexError:
                pass
            
            x_labels_final.append(cx/self.pixel_size_final*2.27)
            y_labels_final.append(cy/self.pixel_size_final*2.27)  
                      
            area.append((radv*radh*np.pi)*(2.27**2))
        
        black=0,0,0
        pink=230,0,230

        mask = cv2.resize(ves_area,(shape0,shape1))     
        mask=Image.fromarray((mask*255).astype('uint8'))        
        mask=ImageOps.grayscale(mask)
        mask=ImageOps.colorize(mask,black,pink)       
        mask=mask.convert('RGBA')
        
        pixeldata = mask.getdata()
        transparent_list=[]
                    
        temp=[]
        for n,pixel in enumerate(pixeldata):
            if pixel[0]>0:
                temp.append(pixel[0])   
            
        for pixel in pixeldata:
            if pixel[0:3]==(0,0,0):
                transparent_list.append((0,0,0,0))
            else: 
                transparent_list.append((pixel[0],pixel[1],pixel[2],30))

        mask.putdata(transparent_list)               

        coordinates=list(zip(x_labels_final,y_labels_final)) 
                   
        self.coordinates=coordinates
        self.mask=mask
        self.area=np.array(area)

    def drawing_ellipse(self,radius_ve,radius_ho):
        cx=20
        cy=20
        ellipse=np.zeros((40,40))
        re_maj=radius_ve
        ri_maj=re_maj-3
        re_min=radius_ho
        ri_min=re_min-3
        ye,xe=np.ogrid[-re_maj:re_maj,-re_min:re_min]
        index_e=xe**2/(re_min**2)+ye**2/(re_maj**2) <=1
        ellipse[cy-re_maj:cy+re_maj,cx-re_min:cx+re_min][index_e]=1
        yi,xi=np.ogrid[-ri_maj:ri_maj,-ri_min:ri_min]
        index_i=xi**2/(ri_min**2)+yi**2/(ri_maj**2) <=1
        ellipse[cy-ri_maj:cy+ri_maj,cx-ri_min:cx+ri_min][index_i]=0    
        return ellipse            
        

    def manual_correction1(self):
        
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
        
        button1=tk.Button(self.master,text='Search file',command=self.manual_correction2,font=('Helvetica', '10'))
        self.canvas1.create_window(400,350,window=button1)
                       
    def manual_correction2(self):

        self.canvas1.destroy()
        
        #getting the image path
        self.master.filepath=filedialog.askopenfilename(initialdir = "/",title = "Select file")
        image_path=self.master.filepath
        if image_path[-9:]=='_mask.tif':
            image_path=image_path[:-9]+'.tif'
        
        splitted=image_path.split('.')
        mask_path=image_path.split('.')[0]+'_mask.tif'
        
        if len(splitted)>2:
            mask_path=image_path.split('.')[0]
            for s in range(len(splitted)-2):
                mask_path=mask_path+'.'+image_path.split('.')[s+1]
            mask_path=mask_path+'_mask.tif'
        
        self.mask_path=mask_path
        
        img=Image.open(image_path)
        self.img_correction=img
            
        list_path=image_path.split('/')
        result_dir=''
        for i in range(len(list_path)-1):
            temp=list_path[i]+'/'
            result_dir=result_dir+temp
               
        sheet_name=list_path[-1]
        splitted=sheet_name.split('.')
        self.sheet_name=sheet_name.split('.')[0]

        if len(splitted)>2:
            for s in range(len(splitted)-2):
                self.sheet_name=self.sheet_name+'.'+sheet_name.split('.')[s+1]
                    
        if (self.results_to_use.get()[-4:]=='xlsx') or (self.results_to_use.get()[-3:]=='xls'):
            excel_filename=self.results_to_use.get()
        else:    
            excel_filename=self.results_to_use.get() +'.xlsx'

        self.xls = pd.ExcelFile(result_dir+excel_filename)
        df_labels = pd.read_excel(self.xls,self.sheet_name,header=0)
        x=[]
        y=[]
        for idx,row in df_labels.iterrows():
            x.append(row['x_values'])
            y.append(row['y_values'])
            
        x=np.array(x)
        y=np.array(y)

        # plotting image and labels
        self.fig=Figure(figsize=(10,8))
        self.ax=self.fig.add_subplot(1,1,1)
        self.ax.imshow(img,cmap='gray') 
        self.ax.scatter(x,y,c='blue',s=4)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
        try:
            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()           
            self.canvas3=FigureCanvasTkAgg(self.fig,master=self.master)        
            toolbar=tkagg.NavigationToolbar2Tk(self.canvas3, self.master)
            self.toolbar.update()
            self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)
       
        except AttributeError:             
            self.canvas3=FigureCanvasTkAgg(self.fig,master=self.master)
            self.toolbar=tkagg.NavigationToolbar2Tk(self.canvas3, self.master)
            self.toolbar.update()
            self.canvas3.get_tk_widget().pack(fill=tk.BOTH,expand=1)            
        
        if len(x) != 1:
            self.original_values=np.array(list(zip(x,y)))
        else:
            self.original_values=np.array([[x,y]])
            
        self.new_values=[]
        self.canvas3.mpl_connect('key_press_event',self.add_or_remove_points)
        self.canvas3.mpl_connect('key_press_event',self.excel_update1)
        
        
    def add_or_remove_points(self,event):
        if event.key == 'a':
    
            new_xy_datapoint=[event.xdata,event.ydata]
            
            if len(self.new_values)==0:
                if len(self.original_values)>0:
                    old_xydata=self.original_values
            else:
                old_xydata=self.new_values
            
            if (len(self.original_values)>0) or (len(self.new_values)>0):
                self.new_values = np.insert(old_xydata,0,new_xy_datapoint,axis=0)  
            else:
                self.new_values=np.array(new_xy_datapoint).reshape(1,-1)
                
            self.ax.scatter(event.xdata,event.ydata,c='blue',s=4)
            self.fig.canvas.draw()
                
        if event.key == 'd':
            
            if (len(self.new_values)==0) and (len(self.original_values)==0):
                pass
            else:
                xy_click=np.array([event.xdata,event.ydata])
    
                if len(self.new_values)==0:
                    old_xydata=self.original_values
                else:
                    old_xydata=self.new_values
                    
                #find false detected vesicles closest to click
                # initiate the search with the position of the first vesicle
                closest_point=old_xydata[0]   
                min_euc=999999
                idx_to_delete=0
                for idx in range(len(old_xydata)):
                    euc_distance=euclidean_distances(old_xydata[idx].reshape(1,-1), xy_click.reshape(1,-1))
                    if euc_distance<min_euc:
                        min_euc=euc_distance
                        closest_point=old_xydata[idx]
                        idx_to_delete=idx
                
                # delete the x,y point corresponding to the false detected vesicle 
                if min_euc< 12:
                    self.new_values = np.delete(old_xydata,idx_to_delete,axis=0)            
                    self.ax.scatter(closest_point[0],closest_point[1],c='red',s=4)
                    self.fig.canvas.draw()            
    
    def excel_update1(self,event):

        # for this update to work it it important to DO NOT EDIT MANUALLY the result excel file
        if (len(self.new_values) >0) and (event.key == 'u'):    

            self.canvas3.get_tk_widget().pack_forget()
            self.toolbar.pack_forget() 
            
            self.canvas1 = tk.Canvas(self.master, width = 800, height = 600,  relief = 'raised')
            self.canvas1.pack()
            
            label2 = tk.Label(self.master, text='Please, remind me the pixel size (1 side!) in nanometer')
            label2.config(font=('helvetica', 12))
            
            self.canvas1.create_window(400, 200, window=label2)
        
            self.pixel_size = tk.Entry (self.master)             
            self.canvas1.create_window(400, 250, window=self.pixel_size)
            
            
            button_update=tk.Button(self.master,text='Click here to update your results',command=self.excel_update2,font=('Helvetica', '10'))
            self.canvas1.create_window(400,350,window=button_update)
        
        else:
            pass           
    
    def excel_update2(self):
        
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
        
        
        self.pixel_size_final=float(self.pixel_size.get())

        with pd.ExcelWriter(self.xls, engine='xlsxwriter') as writer:
            df_all = pd.read_excel(self.xls, sheet_name=None, header=0)    
            
            #calculate distance nearest vesicles again
            x,y=zip(*self.new_values)
            X=np.array([x,y]).T
            euc_distances=euclidean_distances(X, X)

            euc_distances[np.where(euc_distances==0.0)]=10000
            min_distances=[]
            for i in range(euc_distances.shape[0]):
                temp=min(euc_distances[i,:])
                min_distances.append(temp*float(self.pixel_size.get()))    
            
           
            # here calculate area 
            img_corr=self.img_correction.resize((int(self.img_correction.size[0]*self.pixel_size_final/2.27),int(self.img_correction.size[1]*self.pixel_size_final/2.27)))
            np_img_corr=np.array(img_corr)
            
            ves_area=np.zeros((np_img_corr.shape[0],np_img_corr.shape[1]))        
            
            x_sf=np.zeros((len(x)))
            y_sf=np.zeros((len(y)))
            for ii in range(len(x)):
                x_sf[ii]=x[ii]*self.pixel_size_final/2.27
                y_sf[ii]=y[ii]*self.pixel_size_final/2.27
            
            x_labels_final=[]
            y_labels_final=[]
            area=[]
            
            for i in range(len(x_sf)):            
                minimo=2550000
                ho_shift=0
                ve_shift=0
                radv=10
                radh=10
                shift=[-3,-2,-1,0,1,2,3]
         
                for o in range(7):
                    for v in range(7):
                        shift_ho=shift[o]
                        shift_ve=shift[v]
                        snap=np_img_corr[int(y_sf[i]-20+shift_ve):int(y_sf[i]+20+shift_ve),int(x_sf[i]-20+shift_ho):int(x_sf[i]+20+shift_ho)]
                        if snap.shape != (40,40):
                            continue
                        for rv in range(6):
                            for rh in range(6):
                                ring=self.drawing_ellipse(7+rv,7+rh)
                                if (np.abs(rv-rh)<5):
                                    matrice=ring*snap
                                    st_dev=np.std([np.mean(matrice[:10,:10]),np.mean(matrice[10:,:10]),np.mean(matrice[:10,10:]),np.mean(matrice[10:,10:])])
                                    area_ring=np.sum(ring==1)
                                    membrane_value=np.sum(matrice)/area_ring
                                    value_comp=membrane_value+(0.03*st_dev)
                                    if value_comp<minimo:
                                        minimo=membrane_value
                                        ho_shift=shift_ho
                                        ve_shift=shift_ve
                                        radv=7+rv
                                        radh=7+rh

                                        
                cy=int(y[i])
                cx=int(x[i])
                
                ye,xe=np.ogrid[-radv:radv,-radh:radh]
                index_e=xe**2/(radh**2)+ye**2/(radv**2) <=1
                try:
                    ves_area[cy-radv:cy+radv,cx-radh:cx+radh][index_e]=1
                except IndexError:
                    pass
                                    
                x_labels_final.append((x_sf[i])/self.pixel_size_final*2.27)
                y_labels_final.append((y_sf[i])/self.pixel_size_final*2.27)  
                area.append((radv*radh*np.pi)*(2.27**2))
            
            #change values in sheet related to processed image
            if len(x)!=1:
                new_filling=np.array(list(zip(x_labels_final,y_labels_final,min_distances,area)))
            else:
                new_filling=np.array([[x,y,min_distances,area]])
                
            # make mask with update area    
            black=0,0,0
            pink=230,0,230
    
            mask = cv2.resize(ves_area,(self.img_correction.size[0],self.img_correction.size[1]))     
            mask=Image.fromarray((mask*255).astype('uint8'))        
            mask=ImageOps.grayscale(mask)
            mask=ImageOps.colorize(mask,black,pink)       
            mask=mask.convert('RGBA')
            
            pixeldata = mask.getdata()
            transparent_list=[]
                        
            temp=[]
            for n,pixel in enumerate(pixeldata):
                if pixel[0]>0:
                    temp.append(pixel[0])   
                
            for pixel in pixeldata:
                if pixel[0:3]==(0,0,0):
                    transparent_list.append((0,0,0,0))
                else: 
                    transparent_list.append((pixel[0],pixel[1],pixel[2],30))
    
            mask.putdata(transparent_list)                           
            mask.save(self.mask_path)            
            
            #rewrite excel file

            for k in df_all.keys():
                if k=='Summary results':
                    df_ss=df_all[k]
                    df_ss.loc[df_ss['Image name']==self.sheet_name,'Vesicles count']=len(x)
                    df_ss.to_excel(writer,sheet_name=k,index=False)
                    
                elif k == self.sheet_name:
                    df_ss=pd.DataFrame(columns=['x_values','y_values','Distance to nearest vesicle (nm)','Area (nm²)'])
                    for i,e in enumerate(new_filling):
                        df_ss.loc[str(i)]=e
                    df_ss.to_excel(writer,sheet_name=k,index=False)

                else:
                    df_ss=df_all[k]
                    df_ss.to_excel(writer,sheet_name=k,index=False)
            
            #save and close
            writer.close()
            writer.save()
    
        label2 = tk.Label(self.master, text='Update done!')
        label2.config(font=('helvetica', 12))
        self.canvas1.create_window(400, 300, window=label2)
