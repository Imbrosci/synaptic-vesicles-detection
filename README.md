# Synaptic vesicles automated detection

### The present algorithm detects and localizes presynaptic vesicles in electron micrographs.


## Prerequisites 

1)	Anaconda with python 3 (https://www.anaconda.com/distribution/). The present algorithm was tested on python 3.7, 3.8 or 3.9. We recommend to create a virtual environment, with one of these versions. For instance, to create a virtual environment with python 3.9, type: conda create –n env_name anaconda python=3.9. If you already have anaconda with one of the tested version of python, type: conda create -n env_name. To activate the created virtual enviroment type: conda activate env_name (*env_name may be a name of your choice).

2)	PyTorch (used version 1.10). The installation command depends on your Operating System and Compute Platform, to install PyTorch correctly please follow the instruction at: https://pytorch.org/. 

3) Additional required libraries can be installed using the requirements.txt file which can be found in this repository. The installation of these libraries should be done after cloning the repository (see Installation section below). 

## Installation 

Clone or download the following GitHub repository:
https://github.com/Imbrosci/synaptic-vesicles-detection.git.
The file model.pth and model_post.pth, containing the trained first and second classifier, respectively, are too big to be downloaded by git and therefore are replaced by a text file while cloning the repository. In order to download them correctly, after having cloned-downloaded the repository, go to: 
https://github.com/Imbrosci/synaptic-vesicles-detection/blob/master/vesicles_analysis/model.pth and https://github.com/Imbrosci/synaptic-vesicles-detection/blob/master/vesicles_analysis/model_post.pth and click download to download the two files manually. 
Then, replace the model.pth and model_post.pth files downloaded by git with the manually downloaded ones in the folder synaptic-vesicles-detection/vesicles_analysis.  
Finally, install the required libraries listed in requirements.txt. To this end, navigate to the synaptic-vesicles-detection folder (where the requirements.txt file should be located) and type the following command: pip install -r requirements.txt.

## Preliminary step before starting the analysis 

The only required step is to measure the pixel size of the images you want to analyse. To allow vesicle detection on EM images with different magnification and resolution we implemented a step to rescale the images so that a 40x40 pixel window would have the same size as the images we used to train the vesicle classifier (circa 90 x 90 nm). For this step, the experimenter will be asked to provide the pixel size (in nm) of the images to be analyzed. Be aware that providing the wrong pixel size will compromize the results. 

## Starting the analysis

1.	Move the images to be analysed in the folder vesicles_analysis;
2.	Run the script running_analysis.py. This will automatically open a graphical user interface;
3.	Select Analysis > Vesicles detection;
4.	Provide, as requested, an experiment name and the dimension of the pixels (1 side) in nanometer (in other words, how big is each side of one pixel in nanometers) and choose if you want the algorithm to estimate the vesicles area by typing 'y' or 'n' (and not yes or not). If you type a different letter or a different text, the algorithm, by default, is not going to estimate the vesicles area. Be aware that if you decide to include this analysis, the algorithm may take a bit longer to process the images. In this regard, it is worth to mention that the coordinates of each detected vesicle may vary slightly (no more than 6.81 nm) between performing the analysis with or without calculation of the vesicles area. This is because the calculation of the vesicle area offers the possibility to do little adjustments to better find the centre of the detected vesicle.  

If everything works correctly the analysis will start automatically within a few seconds.

## Excel file containing the results

The results will automatically be saved in an excel file which can be found under the folder vesicles_analysis. 
The so generated excel file consists of a summary result sheet with the name and the vesicles count of each analyzed image and a separate sheet for each analysed image containing the following information for each detected vesicle: the x, y coordinates, the distance to the nearest vesicle and the estimated area. 
The name of the excel file will correspond to the name given to the experiment.  
To perform a second round of analysis it is necessary to remove the already analysed images and to add the new images to be analysed in the vesicles_analysis folder. To avoid the overwriting of the excel file containing the results originated with the first round of analysis, it is important to choose a different experiment name for the second round of analysis, so that a different excel file will be generated. 

## Checking the results 

Once the analysis is finished, it is possible to visualize the vesicles detected in each image as follows:

1.	Select Results check> Display detection on image;
2.	Enter the name of the experiment;
3.	Select the analysed image you want to display.  

It is also possible to visualize the vesicle counts, the mean nearest neighbor distances and the estimated vesicles area for all images grouped by different experiments as follows (the graph with the estimated vesicles area will be displayed if this option was chosen in all experiments):

1.	Select Results check > Display graphic results;
2.	Enter the name of the experiment;
3.	Select the folder where the excel file/s with the results is/are located (by default the excel files are generated in the folder vesicles_analysis). 

Finally, if the results are not satisfying, it is possible to manually correct them as follows:

1. Select Results check > Manual correction;
2. Enter the name of the experiment;
3. Select an image (this will display the image and the detected vesicles as blue dots);
4. To add missed vesicles (false negatives): move the cursor to the missed vesicle and press the keyboard button ‘a’, a new blue dot should appear on the cursor position;
5. To remove false positives: move the cursor to the erroneously detected vesicle and press the keyboard button ‘d’, the blue dot should turn into red;
6. Once corrections are finished press the keyboard button 'u' to update the results (at this stage, you will be asked again to provide the pixel size of the image). Once the update is done, the excel file will contain the corrected results. Be aware that if the corrections were done on an image from which the vesicles area were estimated, the update may take some time.

## Final notes

1.	We strongly recommend to run the analysis on a computer equipped with a graphics processing unit (GPU). This will drastically increase the speed of the analysis. We also recommend to use a computer with a Linux or Windows Operating System as tkinter may crash on Mac.

2.	During analysis the program will generate a semi-transparent pink mask for each image, corresponding to the estimated vesicles area. The mask will automatically be named ‘imagename_mask’. Avoid naming the images that you want to analyse with a name that terminate with ‘mask’ (extension excluded), otherwise the program will treat the image as a mask and will skip the analysis for that image.

3. We recommend to use 8-bit images, otherwise problems by displaying the results may occur.

4.	The files first_classifier_training.py and second_classifier_training.py are not needed for running the analysis but they could be helpful in case one decides to re-train the model with her/his own data. 

5. The datasets used to train and test the two classifiers and to evaluate the performance of the algorithm in its entirety are publicly available at the following doi: https://doi.org/10.12751/g-node.s4bsnu. 

6.	The algorithm presented here performed well on images obtained using an EM900 (Zeiss), a JEM-1011 (JEOL), and a Tecnai G20 (Thermo Fisher Scientific) transmission electron microscopes operating at 80-120 kV. The performance on images obtained with a scanning electron microscope was slightly inferior. If needed, you should consider the possibility to create a separate training dataset to customize the model for the single need. To this end we provide the source code containing the employed classifiers (CNNs_GaussianNoiseAdder.py) as well as the source codes to train the two classifiers (first_classifier_training.py and second_classifier_training.py). For more information on how to train the model with your own data and on how to create a suited dataset you can contact us at barbara.imbrosci@googlemail.com or marta.orlando@charite.de. 

## Reporting issues

Issues can be reported in the issue tracker of this repository.
