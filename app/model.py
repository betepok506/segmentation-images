import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score)
import cPickle
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import rasterio


class SegmentationRandomForestClassifier:
    def __init__(self, scaler=StandardScaler(),
                 clf=RandomForestClassifier(n_jobs=-1, n_estimators=10)):
        self.scaler = scaler
        self.model = clf

    def predict(self, images):
        result = []
        if not isinstance(images, list):
            images = [images]

        for image in images:
            df = {
                "red": [],
                "green": []
            }

            for row in range(image.shape[1]):
                for col in range(image.shape[2]):
                    df["red"].append(image[0][row][col])
                    df["green"].append(image[1][row][col])

            df = pd.DataFrame(df)
            df = self.scaler.transform(df)
            y_pred = self.model.predict(df)
            pred_mask = y_pred.reshape((256, 256))
            result.append(pred_mask)

    def train(self, X, y, path_to_save: str):
        self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.save(path_to_save)

    def save(self, path_to_save: str):
        path_to_save = os.path.join(path_to_save, "output")
        os.makedirs(os.path.join(path_to_save, "output"), exist_ok=True)
        path_to_save_classifier = os.path.join(path_to_save, 'classifier.pkl')
        with open(path_to_save_classifier, 'wb') as fid:
            cPickle.dump(self.model, fid)

        path_to_save_scaler = os.path.join(path_to_save, 'scaler.pkl')
        with open(path_to_save_scaler, 'wb') as fid:
            cPickle.dump(self.scaler, fid)

    def load(self, path_to_load: str):
        path_to_load_classifier = os.path.join(path_to_load, 'classifier.pkl')
        with open(path_to_load_classifier, 'rb') as fid:
            self.model = cPickle.load(fid)

        path_to_load_scaler = os.path.join(path_to_load, 'scaler.pkl')
        with open(path_to_load_scaler, 'rb') as fid:
            self.scaler = cPickle.load(fid)

    def evaluate(self, X, y):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return {
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred)
        }

    @staticmethod
    def _read_nvdi(path_to_image: str):
        index = rasterio.open(path_to_image)
        img = index.read((7, 5, 3))
        return index, img

    @staticmethod
    def _read_mask(path_to_mask: str):
        mask = cv.imread(path_to_mask, cv.IMREAD_GRAYSCALE)
        return mask

    @staticmethod
    def dataset_load(path_to_data: str,
                     drop_columns: list = [],
                     label="labels"):
        drop_columns.append(label)
        data = pd.read_csv(path_to_data)
        data = data.drop(drop_columns, axis=1)
        return data

    def dataset_create(self,
                       path_to_images: str,
                       paths_to_masks: str,
                       label="labels",
                       mask_algorithm='Kumar-Roy'):
        d = {
            "red": [],
            "green": [],
            label: []
        }

        for ind, path_to_mask in tqdm(enumerate(paths_to_masks)):
            name_file_image = os.path.basename(path_to_mask).split('.')[0].replace(f"_{mask_algorithm}", '')
            path_to_image = os.path.join(path_to_images, name_file_image + ".tif")
            index, image = self._read_nvdi(path_to_image)

            _, h, w = image.shape
            true_mask = self._read_mask(path_to_mask)

            for row in range(image.shape[1]):
                for col in range(image.shape[2]):
                    d["red"].append(image[0][row][col])
                    d["green"].append(image[1][row][col])
                    d[label].append(true_mask[row][col])

        data = pd.DataFrame(d)
        return data
