# Synaptic vesicles automated detection

### The present algorithm detects and localizes presynaptic vesicles in 2D electron micrographs.


## Prerequisites 

1)	Anaconda with python 3 (https://www.anaconda.com/distribution/). We recommend to use python 3.6 or 3.7. In case your python version is different you may want to create and activate a virtual environment. For instance, to create a virtual environment with python 3.6, type: conda create –n env_name anaconda python=3.6. To activate the created virtual enviroment type: conda activate env_name or source activate env_name 
(*env_name may be a name of your choice)

2)	opencv, to install opencv you can use one of the following commands:
 * conda install –c https://conda.binstar.org/menpo opencv
 * conda install –c conda-forge opencv 

3)	xlsxwriter,to install xlswriter you can use the following command: 
* conda install xlsxwriter

4)	pillow, to install pillow you can use the following command: 
* conda install pillow

5)	pytorch torchvision, to install pytorch torchvision correctly follow the instruction at: https://pytorch.org/ 

## Installation 

Clone or download the following GitHub repository:
https://github.com/Imbrosci/synaptic-vesicles-detection.git.
The file model.pth, containing the weights of the trained model, is too big to be download by git and therefore is replaced by a text file. In order to download it correctly, after having cloned-downloaded the repository, go to: 
https://github.com/Imbrosci/synaptic-vesicles-detection/blob/master/vesicles_analysis/model.pth 
and click download to download model.pth manually. 
Then, replace, in the directory synaptic-vesicles-detection/vesicles_analysis, the model.pth file downloaded by git with the manually downloaded one. 


## Preliminary step before starting the analysis 

Measuring the pixel size. To allow vesicle detection on EM images with different magnification and resolution we implemented a step to rescale the images so that a 40x40 pixel window would have the same size as the images we used to train the vesicle classifier (circa 90 x 90 nm). For this step, the experimenter will be asked to provide the pixel size (in nm) of the images to be analyzed.

## Starting the analysis

1.	Move the images to be analysed in the directory vesicles_analysis;
2.	Run the script running analysis. This will automatically open a graphical user interface;
3.	Select Analysis > Vesicles detection;
4.	Provide, as requested, an experiment name and the dimension of the pixels in nanometer.

If everything works correctly the analysis will start automatically within a few seconds.

## Excel file containing the results

The results will automatically be saved in an excel file which can be found under the directory vesicles_analysis. 
The so generated excel file should consist of a summary result sheet with name and the vesicles count for each analyzed image and a separate sheet for each image containing the following information: the vesicle position, the distance to the nearest vesicle and the area for each detected vesicle. 
The name of the excel file will correspond to the name given to the experiment.  
To perform a second round of analysis it is necessary to remove the already analysed images and to add the images to be analysed in the vesicles_analysis directory. To avoid the overwriting of the excel file containing the results originated with the first round of analysis, it is important to choose a different experiment name for the second round of analysis, so that a different excel file will be generated. 

## Checking the results 

Once the analysis is finished, it is possible to visualize the vesicles detected in each image as following:

1.	Select Results_check> Display detection on image;
2.	Enter the name of the experiment;
3.	Select the analysed image you want to display.  

It is also possible to visualize vesicle counts, mean nearest neighbor distances and vesicles area for all images grouped by different experiments as following:

1.	Select Results_check > Display graphic results;
2.	Enter the name of the experiment;
3.	Select the directory where the excel file/s with the results are located (by default the excel files are generated in the directory vesicles_analysis). 

Finally, if results are not satisfying, it is possible to manually correct them as following:

1. Select Results_check > Manual correction;
2. Enter the name of the experiment;
3. Select an image (this will display the image and the detected vesicles as white dots);
4. To add missed vesicles (false negatives): move the cursor to the missed vesicle and press the keyboard button ‘A’, a white dot should appear on the cursor position;
5. To remove false positives: move the cursor to the erroneously detected vesicle and press the keyboard button ‘D’, the white dot should turn into red;
6. Once corrections are finished press the keyboard button 'U' to update the results (at this stage, you will be asked again to provide the pixel size of the image). Once the update is done, the excel file will contain the corrected results.

## Final notes

1.	We strongly recommend to run the analysis on a computer equipped with a graphics processing unit (GPU). This will drastically increase the speed of the analysis. Also, note that tkinter crashes macOS in the macOS update 10.14.4 - 10.14.6; This is a macOS bug that we cannot solve.

2.	During analysis the program will generate a semi-transparent pink mask for each image, corresponding to the estimated vesicles area. The mask will automatically be named ‘imagename_mask’. Avoid naming the images that you want to analyse with a name that terminate with ‘mask’ (extension excluded), otherwise the program will treat the image as a mask and will skip the analysis for that image.

3. We recommend to use 8-bit images, otherwise problems by displaying the results may occur.

4. In the folder training-testing-images we provide the images we used for creating patches to train the two netowrks we used  (model and model_post) and the images used for final testing. 

5.	The files classifier_training.py, post_classifier_training.py and im_convered.py are not needed for running the analysis but they could be helpful in case one decides to re-train the model with her/his own data. In this case the training and validation dataset should be saved in a folder/subfolder data> train and data>test, respectively. 

6.	The here presented algorithm performed well on images obtained using an EM900 (Zeiss), a JEM-1011 (JEOL), and a Tecnai G20 (Thermo Fisher Scientific) transmission electron microscopes operating at 80-120 kV. The performance on images obtained with a scanning electron microscope was slightly inferior. If needed you should consider the possibility to create a separate training dataset to customize the model for the single need. To this end we provide the source code containing the employed classifiers (vesicle_classifier.py)  as well as the source codes to train the classifiers (classifier_training.py and post_classifier_training.py). For more information on how to train the model with your own data and on how to create a suited dataset you can contact us at barbara.imbrosci@googlemail.com or marta.orlando@charite.de. 

## Reporting issues

Issues can be reported in the issue tracker of this repository.
