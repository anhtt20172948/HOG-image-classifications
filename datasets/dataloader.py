import os
from abc import abstractmethod
from glob import glob
from utils.utils import *
from feature_extractor.hog import HOG_feature_extractor
from feature_extractor.configs import *
from tqdm import tqdm


class CIFA10Dataloader(object):
    def __init__(self, path):
        self.datasets = path
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.feature_extractor = HOG_feature_extractor(HOG_CONFIG)

    def extract_feature(self, save=None, overwrite=False):
        """

        :return: train_data: numpy array of train features
                train_labels: numpy array of train labels
                test_data: numpy array of test features
                test_labels: numpy array of test labels

        """
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for file in glob(os.path.join(self.datasets, '*')):
            print("PROCESSING CIFA 10 Datasets: {}".format(os.path.basename(file)))
            name = os.path.basename(file)
            data = unpickle(file)
            images = data[b'data']
            labels = data[b'labels']
            for i, img in enumerate(tqdm(images)):
                single_img = np.array(img)
                single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
                fd = self.feature_extractor.get_feature(single_img_reshaped)
                if 'data' in name:
                    train_data.append(fd)
                    train_labels.append(labels[i])
                else:
                    test_data.append(fd)
                    test_labels.append(labels[i])

        self.train_data = np.asarray(train_data)
        self.train_labels = np.asarray(train_labels)
        self.test_data = np.asarray(test_data)
        self.test_labels = np.asarray(test_labels)
        if save is not None:
            if not os.path.isdir(save):
                os.mkdir(save)
                self.save(save, overwrite=overwrite)
            self.save(save, overwrite=overwrite)

    def save(self, path: str, overwrite=False):
        """

        :param overwrite: overwrite feature
        :param path: path to save train and test feature
        :return:
        """
        to_save_train = {
            'data': self.train_data,
            'labels': self.train_labels
        }
        to_save_test = {
            'data': self.test_data,
            'labels': self.test_labels
        }
        train_file = os.path.join(path, 'train.pkl')
        test_file = os.path.join(path, 'test.pkl')
        if os.path.isfile(train_file):
            if overwrite:
                savepickle(train_file, to_save_train)
                print("OVER WRITE TRAIN FILE")
            else:
                print("NOT SAVE, Train Feature already exists , PLEASE ENABLE overwrite")
                if yes_or_no("Do you want to enable overwrite: "):
                    savepickle(train_file, to_save_train)

        else:
            print("SAVING TRAIN FEATURES")
            savepickle(train_file, to_save_train)
            print("SAVED TRAIN FEATURES")

        if os.path.isfile(test_file):
            if overwrite:
                savepickle(test_file, to_save_test)
                print("OVER WRITE TEST FILE")
            else:
                print("NOT SAVE,  Test Feature already exists , PLEASE ENABLE overwrite")
                if yes_or_no("Do you want to enable overwrite: "):
                    savepickle(test_file, to_save_test)
        else:
            print("SAVING TEST FEATURES")
            savepickle(test_file, to_save_test)
            print("SAVED TEST FEATURES")

    @abstractmethod
    def load(self, path: str):
        """

        :param path:
        :return: data: numpy arrays contain features of images
                labels: numpy arrays of labels
        """
        with open(path, 'rb') as f:
            datas = pickle.load(f)
            data = datas['data']
            labels = datas['labels']
        return data, labels
