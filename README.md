# Morph-UNet: An Approach Towards Medical Image Segmentation using Novel Multiscale Trainable  Morphological Modules
## **Dataset**:

**Skin lesion segmentation**:

- **ISIC2017**: https://challenge.isic-archive.com/data/#2017

* **ISIC2018**: https://challenge.isic-archive.com/data/#2018

+ **HAM10000**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
  
**Breast Tumor Segmentation**:

- **BUSI**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

+ **UDIAT**: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8003418 (https://www.kaggle.com/datasets/jarintasnim090/udiat-data)

**Gland Segmentation**:

- **GlaS**: https://academictorrents.com/details/208814dd113c2b0a242e74e832ccac28fcff74e5

**Nuceli segmentation**:

- **MonuSeg** (Binary): https://monuseg.grand-challenge.org/Data/
+ **PanNuke** (Multiclass): https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke


## **Inputs data structure**:
   - The inputs data for ISIC2017, BUSI, ISIC2018, UDIAT, HAM10000 dataset must be structred in folders as instructed by https://github.com/jeya-maria-jose/UNeXt-pytorch

- run train_ISIC_BUSI.py and change the -arch arguements according to the model (UNext, UNet, Morph-UNet, MALUNet etc)
  



