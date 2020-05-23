# Synaptic vesicles automatic detection

### The present algorithm detects and localizes presynaptic vesicles in 2D electron micrographs.


## Prerequisites 

1)	Anaconda with python 3 (https://www.anaconda.com/distribution/)
2)	Due to incompatibility issues we recommend to use python 3.6. In case your python version is different you may want to create and activate a virtual environment.'''
Creating a virtual environment: conda create –n env_name anaconda python=3.6 
Activating the virtual environment: conda activate env_name or source activate env_name 
(*env_name may be a name of your choice)
3)	opencv
you can try one of the following commands:
conda install –c https://conda.binstar.org/menpo opencv
conda install –c conda-forge opencv 
4)	xlsxwriter, you can use the following command: conda install xlsxwriter
5)	pillow, you can use the following command: conda install pillow
6)	pytorch torchvision (follow the instruction at: https://pytorch.org/ )

## Installation 

Clone or download the following GitHub repository:
https://github.com/Imbrosci/synaptic-vesicles-detection.git
The file model.pth, containing the weights of the trained model, is too big to be download by git and therefore is replaced by a text file. In order to download it correctly, after having cloned-downloaded the repository, go to: 
https://github.com/Imbrosci/synaptic-vesicles-detection/blob/master/vesicles_analysis/model.pth 
and click download to download model.pth manually. 
Then, replace, in the directory synaptic-vesicles-detection/vesicles_analysis, the model.pth file downloaded by git with the manually downloaded one. 

## Preliminary steps before starting the analysis

Creating a mask. To guarantee the detection of vesicles within a presynaptic terminal or within another subcellular structure of interest, the experimenter should create a black mask on all regions of the EM image outside the area of interest, including the plasma membrane. Furthermore, we recommend to create a black mask on mitochondria, multivesicular bodies and on membrane enclosed postsynaptic regions. This can easily be done with the program Fiji1. With this initial step, we noticed an optimal efficiency of the subsequent automated analysis. 
Measuring the pixel size. To allow vesicle detection on EM images with different magnification and resolution we implemented a step to rescale the image so that a 40x40 pixel window would have the same size as the images we used to train the vesicle classifier (91.08 x 91.08 nm). For this step, the experimenter will be asked to provide the pixel size of the images to be analyzed.

1 Schindelin, J., Arganda-Carreras, I., Frise, E., Kaynig, V., Longair, M., Pietzsch, T., Preibisch, S., Rueden, C., Saalfeld, S., Schmid, B., Tinevez, J.-Y., White, D.J., Hartenstein, V., Eliceiri, K., Tomancak, P., Cardona, A., 2012. Fiji: an open-source platform for biological-image analysis. Nat. Methods 9, 676–682. https://doi.org/10.1038/nmeth.2019

## Starting the analysis

1.	Move the images to be analysed in the directory vesicles_analysis;
2.	Run the script running analysis. This will automatically open a graphical user interface;
3.	Select Analysis > Vesicles detection;
4.	Provide, as requested, an experiment name and the dimension of the pixels in nanometer.

If everything works correctly the analysis will start automatically within a few seconds.
The results will automatically be saved in an excel file which can be found under the directory vesicles_analysis. The name of the excel file will correspond to the name given to the experiment.  
To perform a second round of analysis it is necessary to remove the already analysed images and to add the images to be analysed in the vesicles_analysis directory. To do not overwrite the excel file with the results, originated with the first round of analysis, it is important to choose a different experiment name. 

## Displaying the results 

Once the analysis is finished, it is possible to visualize the vesicles detected in each image as following:

1.	Select Display> Display detection on image;
2.	Enter the name of the experiment;
3.	Select the analysed image you want to display.  

It is also possible to visualize vesicle counts and mean nearest neighbor distances for all images grouped by different experiments as following:

1.	Select Display> Display graphic results;
2.	Enter the name of the experiment;
3.	Select the directory where the excel file/s with the results are located (by default the excel files are generated in the directory vesicles_analysis). 

## Final notes

1.	We strongly recommend to run the analysis on a computer equipped with a graphics processing unit (GPU). The analysis of a single image of circa 4x4 micrometer should last only a few minutes with a GPU while it can last more than 30 min without a GPU.

2.	During analysis the program will generate a mask for each image. The mask will automatically be named ‘imagename_mask’. To avoid that the program will treat an image as a mask and therefore skip the analysis for that image, it is important that the name of the images to analyse does not terminate with ‘mask’ (extension excluded). 

3.	The files classifier_training.py and im_convered.py are not needed for running the analysis but they could be helpful in case one decides to re-train the model with her/his own data. In this case the training and validation dataset should be saved in a folder/subfolder data> train and data>test, respectively. 


4.	The here presented algorithm was tested on images obtained using an EM900 (Zeiss) and a Tecnai G20 (Thermo Fisher Scientific) transmission electron microscopes operating at 80-120 kV. We cannot guarantee its performance on images obtained for example from scanning electron microscopes. In the unlucky case that it will not perform equally well on images acquired with a different system we would consider the possibility to create a separate training dataset to customize the model for the single need. 

5.	In case of any problems with the execution of the algorithm you can contact us at: barbara.imbrosci@charite.de
