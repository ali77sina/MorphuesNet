This folder contains the scripts to train the network on the Dreem Open Datasets (DOD) with the healthy participants. 

1. Download.py: Downloads the DOD-H dataset into the current directory. (you need to setup correct AWS credentials to download these files, or you can download them manually from the "dreem-dod-h" AWS S3 bucker)


2. dreemRead.py: Functions to extract data from DOD files, including labels and raw time series signals. Here, you can change what channel will be used for classification. for 1v1 comparison, F3-M2 was used in the paper.


3. nn_model.py: the script containing the functions creating the ResNet CNN and the sequence learner model. 


4. trainer.py: training the models, as described in the paper.


**Desciption**
The CNN is first trained on the data. The data is downsampled to 100Hz, and band-passed between 0.5 and 40 Hz. After the training of the CNN, it is quantized and used to generate the Logits. These are then used to train the sequence learner. sequence lengths were set at 12, but it can be changed.



![image](https://github.com/ali77sina/MorphuesNet/assets/54308350/43a19abf-d638-4684-bcd6-dc1c552192f9)


Fig. 1: example output of the 2 models on fold number 2. figure shows the model's performance on the test patient (the one left out of train-val process)
