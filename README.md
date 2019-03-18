# lesion-GAN
Code for "Towards Interpretable Skin Lesion Classification with Deep Learning Models"

# Link to web interface
http://3.87.210.164:8081/

# HAM10000 Dataset
The dataset used for this project was the HAM10000 dataset, accessible at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

For our purposes, we used the 3 most prominent skin lesion classes.

# Workflow
The "ISIC2018..." and "Processed Images" folders are placeholders for the dataset. After downloading the dataset, it should be placed in the first folder, and processed into the second. Throughout the code, tweaks will have to be made to the path and filenames to run on your machine.

The bulk of the project is in the numbered files, from 1 through 8, where the files should be ran in order. While running the files to convert the images into an numpy-compressed file, please pay attention to filenames.

The exception is the missing "7" file, which represents the CNN in the cnn_notebooks folder. While training the CNN (preferrably with a GPU), the balancedAccuracy.py file is needed to track accuracy during training, though it is not exactly accurate when training and averaging minibatches.

Additionally, after training the CNN (file "7") or multiple CNNs, the predictions of the CNNs on the data can be saved while it is being run on a virtual GPU through the getPredictions.ipynb file in the misc_scripts folder.

# Important Variable References in code
"imageList" refers to the training set (roughly 64% of the modified dataset)

"imageValList" refers to the validation set (80% of the modified dataset in conjunction with imageList)

"testList" refers to the set-aside, testing set (remaining 20% of the modified dataset)
