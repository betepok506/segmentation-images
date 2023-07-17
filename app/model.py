import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score)
import pickle
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
            pred_mask[pred_mask == 1] = 255
            result.append(Image.fromarray(pred_mask.astype('uint8'), 'L'))
        return result

    def train(self, X, y, path_to_save: str):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.save(path_to_save)

    def save(self, path_to_save: str):
        path_to_save = os.path.join(path_to_save, "output")
        os.makedirs(os.path.join(path_to_save, "output"), exist_ok=True)
        path_to_save_classifier = os.path.join(path_to_save, 'classifier.pkl')
        with open(path_to_save_classifier, 'wb') as fid:
            pickle.dump(self.model, fid)

        path_to_save_scaler = os.path.join(path_to_save, 'scaler.pkl')
        with open(path_to_save_scaler, 'wb') as fid:
            pickle.dump(self.scaler, fid)

    def load(self, path_to_load: str):
        path_to_load_classifier = os.path.join(path_to_load, 'classifier.pkl')
        with open(path_to_load_classifier, 'rb') as fid:
            self.model = pickle.load(fid)

        path_to_load_scaler = os.path.join(path_to_load, 'scaler.pkl')
        with open(path_to_load_scaler, 'rb') as fid:
            self.scaler = pickle.load(fid)

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
        img = index.read((1, 2, 3))
        return index, img

    @staticmethod
    def _read_mask(path_to_mask: str):
        mask = cv.imread(path_to_mask, cv.IMREAD_GRAYSCALE)
        return mask

    @staticmethod
    def dataset_load(path_to_data: str,
                     drop_columns: list = [],
                     label="labels"):

        X = pd.read_csv(os.path.join(path_to_data, 'x.csv'))
        y = pd.read_csv(os.path.join(path_to_data, 'y.csv'))
        X = X.drop(drop_columns, axis=1)
        return X, y

    @staticmethod
    def dataset_save(x, y, path_to_save: str):
        os.makedirs(path_to_save, exist_ok=True)
        x.to_csv(os.path.join(path_to_save, "x.csv"), index=False)
        y.to_csv(os.path.join(path_to_save, "y.csv"), index=False)

    @staticmethod
    def dataset_balancing(X, y, label="labels"):
        ones = y[y[label] == 1]
        zeros = y[y[label] == 0]

        x_zeros = X.iloc[zeros.index]
        x_ones = X.iloc[ones.index]
        x = pd.concat([x_zeros.sample(min(len(zeros), len(ones))), x_ones.sample(min(len(zeros), len(ones)))])

        return x, y.iloc[x.index]

    def dataset_create(self,
                       path_to_images: str,
                       path_to_masks: str,
                       max_images: int = 500,
                       label="labels",
                       mask_algorithm='Kumar-Roy'):
        x = pd.DataFrame({"red": [], 'green': []})

        y = pd.DataFrame({
            label: []
        })

        for ind, path_to_mask in tqdm(enumerate(os.listdir(path_to_masks)[:max_images]),
                                      total=len(os.listdir(path_to_masks)[:max_images])):
            name_file_image = os.path.basename(path_to_mask).split('.')[0].replace(f"_{mask_algorithm}", '')
            path_to_image = os.path.join(path_to_images, name_file_image + ".tif.png")
            index, image = self._read_nvdi(path_to_image)

            _, h, w = image.shape
            true_mask = self._read_mask(os.path.join(path_to_masks, path_to_mask))
            tmp_dx = {
                "red": [],
                "green": [],
            }

            tmp_dy = {
                label: []
            }
            for row in range(image.shape[1]):
                for col in range(image.shape[2]):
                    tmp_dx["red"].append(image[0][row][col])
                    tmp_dx["green"].append(image[1][row][col])
                    tmp_dy[label].append(true_mask[row][col])
            tmp_dx = pd.DataFrame(tmp_dx)
            tmp_dy = pd.DataFrame(tmp_dy)

            tmp_dx, tmp_dy = self.dataset_balancing(tmp_dx, tmp_dy)

            x = pd.concat([x, tmp_dx])
            y = pd.concat([y, tmp_dy])
        return x, y


if __name__ == "__main__":
    model = SegmentationRandomForestClassifier()
    # # x, y = model.dataset_create("D:\\projects_andrey\\datasets\\segmentations\\landsat8\\images\\train",
    # #                             "D:\\projects_andrey\\datasets\\segmentations\\landsat8\\masks\\train")
    # # model.dataset_save(x, y, "data")
    # X, y = model.dataset_load('./data')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # model.train(X_train, y_train, "./models")
    # print(model.evaluate(X_test, y_test))


    ##################
    model.load('./models/output/')
    ind , img = model._read_nvdi("D:\\projects_andrey\\datasets\\segmentations\\landsat8\\images\\train//LC08_L1GT_008064_20200813_20200813_01_RT_p00431.tif.png")

    Image.fromarray(img.transpose((1,2,0)).astype('uint8'), 'RGB').show()
    res = model.predict(img)
    res[0].show()
    # cv.imshow("test", res[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print(7)

