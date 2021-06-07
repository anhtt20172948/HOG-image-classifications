import argparse
import pickle
from datasets.dataloader import CIFA10Dataloader
from glob import glob
import os
from models.SVM import SVM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SVM', action='store_true', default=True, help='Linear SVM')
    parser.add_argument('--dataset', type=str, default='datasets/cifar-10-batches-py', help='path to dataset')
    parser.add_argument('--features', type=str, default='features', help='path to folder where to save features')
    parser.add_argument('--extract', action='store_true', default=False, help='extract features from datasets')
    opt = parser.parse_args()
    print(opt)
    features = opt.features
    ciFa10Dataloader = CIFA10Dataloader(opt.dataset)
    if opt.extract:
        print("extracting features from CIFA10 Datasets ")
        ciFa10Dataloader.extract_feature(save=features, overwrite=False)
        print("Features saved: {}".format(features))

    # Load features train and test to train classifications model
    for feature_path in glob(os.path.join(features, '*.pkl')):
        if 'train' in os.path.basename(feature_path):
            train_data, train_labels = ciFa10Dataloader.load(feature_path)
        else:
            test_data, test_labels = ciFa10Dataloader.load(feature_path)

    if opt.SVM:
        clf = SVM()
        print("Start training SVM")
        clf.feed_forward(train_data, train_labels)
