# Steps to implement our algorithm:

Caffe implementation of [A Multi-task Deep Network for Person Re-identification](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14313).

## 1.Build Caffe:
Setup and Compile a Caffe version with triplet loss layer and norm layer. Norm layer is just a normalization operation for fully connected layers.
Details can be found for Caffe Website (http://caffe.berkeleyvision.org/).

## 2.Using BuildData file:
Select a dataset XX and build the training data with files in "BuildData". (Dataset includes: cuhk03,CUHK01,CUHK01half,VIPeR,iLIDS,PRID)
a.Run "BuildData_XX.py" or "BuildAugData_XX.py" to produce the list.
b.Run "create_lmdb_triplet.sh" or "create_lmdb_triplet_aug.sh" to convert the list into lmdb data for Caffe training.

Note:In "BuildData_XX.py", there are two valuation lists in each dataset, one is made by "BuildRankFile()" for classification valuation (MTDnet,MTDnet-cls,MTDnet-cross) and another is made by "BuildTripleFile()" for ranking valuation (MTDnet-rnk).

## 3.Using Nets file:
Select a net in "Nets" and begin training. (Models includes: MTDnet,MTDnet-cls,MTDnet-rnk,MTDnet-cross)
a.Change the srouce path "XXData/XXX" in "train_val.prototxt" with the dataset selected for training.
b.Run "train_finetune_cuhk03.sh" for cuhk03 dataset, run "train_finetune_others.sh" for others.

Note:For MTDnet-cls, you can use the MTDnet instead, as long as changing the loss_weight of triplet loss in train_val.prototxt into 0.
     For MTDtrp, the whole training process is the same as MTDnet.
     For the aug experiment, the training data should be changed from "XXX_train_lmdb_pos1/pos2/neg"to "XXX_trainaug_lmdb_pos1/pos2/neg".

## 4.Using ComputeAccuracy file:
Select a trained model and Output the corresponding accuracy under CMC.
a.Run the "flipimages.py" to produce mirror images for testing sets. Run the "flipimages_iLIDS.py" instead if the sets from iLIDS.
b.Different nets using different computeaccuracy files:
Run "ComputeAccuracy_XX.py" for MTDnet/MTDnet-cls/MTDnet-cross results.
Run "ExtractFeature2Mat_XX.py" for MTDtrp/MTDnet-rnk features, then run "GetAccuracyfromFeatures.py" to get the results.

## 5.TrainedModels
All the trained models in our experiments are listed in "TrainedModel" file, which is available for download if necessary.