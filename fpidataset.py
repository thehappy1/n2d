import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2

class Fpidataset():
    # Constructor
    def __init__(self):

        #self.df = self.df[self.df['fold'] == fold]

        #if transform is not None:
            #transform = torchvision.transforms.Compose([
                #torchvision.transforms.Resize((224, 224)),
                #torchvision.transforms.ToTensor()
            #])
        #self.transform = transform

        self.df = pd.read_csv('data/styles.csv', error_bad_lines=False)
        self.df['image_path'] = self.df.apply(lambda x: os.path.join("data\images", str(x.id) + ".jpg"), axis=1)

        mapper = {}
        for i, cat in enumerate(list(self.df.articleType.unique())):
            mapper[cat] = i
        print(mapper)
        self.df['targets'] = self.df.articleType.map(mapper)

    # Get the length
    def __len__(self):
        return len(self.df)

    def load_data(self):

        self.x_train, self.y_train = self.get_i_items(self.df, 800, train=True)
        self.x_test, self.y_test = self.get_i_items(self.df, 200, train=False)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_i_items(self, df, i, train):
        # get i items of each condition

        # calculate classes with more than 1000 items
        temp = df.targets.value_counts().sort_values(ascending=False)[:10].index.tolist()
        df_temp = df[df["targets"].isin(temp)]

        x_data = []
        y_data = []

        if train==True:
            for label in temp:

                temp_labels = df_temp[df_temp.targets == label]
                train_temp = temp_labels[:i]

                y_data.append(train_temp["targets"].to_list())

                for element in train_temp.image_path:
                    img = cv2.imread(element)
                    x_data.append(img)

                #image normalization fehlt
                print("Anzahl x_train items bei ", label, " :", len(x_data))
                print(" ")
        else:
            for label in temp:

                test_temp = df_temp[df_temp.targets == label]
                test_temp = test_temp[800:1000]

                y_data.append(test_temp["targets"].to_list())

                for element in test_temp.image_path:
                    img = cv2.imread(element)
                    x_data.append(img)

                print("Anzahl x_test items bei ", label, " :", len(x_data))
                print(" ")

        print("y_data: ",y_data)
        return x_data, y_data

