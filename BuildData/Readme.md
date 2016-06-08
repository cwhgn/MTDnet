#Normal Data
##BuildData_XX.py
BuildData_XX.py has two purposes. First, split the XX dataset into three subsets: Train, Val and Test. The images of three sets are saved in three folders respectively. The val set is randomly selected from the datasets. Second, build the training list and the valuation list based on the train and val sets for Caffe training. Note that the val and test sets are the same in datasets except cuhk03. 

FYI,CUHK01 means CUHK01(testID=100), CUHK01half means CUHK01(testID=486), CUHK01 and CUHK01half share the same BuildData_CUHK01.py.

##create_lmdb_triplet.sh
create_lmdb_triplet.sh is used to convert the training and valuation list (.txt) into lmdb format for Caffe training. "XX" can be replace by the name of the datasets including: cuhk03,CUHK01,CUHK01half,iLIDS,VIPeR and PRID.

##RecordedValdata_XX.npy
RecordedValdata_XX.npy are files which record the ids of people used as val data in our experiments, which makes the readers can reproduce our results. 
The format of the Valdata_list_XX.npy are listed below:

Valdata_list_cuhk03.npy (Shape=100*2):
1st Column is Cams(1-3); 2nd Column is IDs.

Valdata_list_CUHK01.npy (Shape=100*1):
1st Column is IDs.

Valdata_list_CUHK01half.npy (Shape=486*1):
1st Column is IDs.

Valdata_list_iLIDS.npy (Shape=59*2):
1st Column is IDs; 2nd Column is the image number according to each ID.

Valdata_list_VIPeR.npy (Shape=316*3):
1st Column is IDs; 2nd Column is the image name in Cam A; 3rd Column is the image name in Cam B.

Valdata_list_PRID.npy (Shape=100*1):
1st Column is IDs.

#Aug Data for the pooled dataset
##BuildAugData_XX.py
BuildAugData_XX.py is used to produce the pooled datasets for the experiments in our supplementary materials. The whole process is similar as BuildData_XX.py, besides adding CUHK03 data into the small XX dataset.

##create_lmdb_triplet_aug.sh
create_lmdb_triplet.sh is used to convert the augmented training list (.txt) into lmdb format for Caffe training. "XX" can be replace by the name of the datasets including: cuhk03,CUHK01,CUHK01half,iLIDS,VIPeR and PRID.

#Auxiliary Data for cross-domain
##BuildCrossData_cuhk03.py
BuildCrossData_cuhk03.py is used to produce the auxiliary from cuhk03 dataset to help other small datasets.

##create_lmdb_forcross.sh
create_lmdb_forcross.sh is used to convert the auxiliary training list (.txt) into lmdb format for Caffe training.
