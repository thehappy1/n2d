import os
import pandas as pd
import numpy as np
import cv2


class Fpidataset():
    # Constructor
    def __init__(self):

        self.df = pd.read_csv('styles.csv', error_bad_lines=False)
        self.df['image_path'] = self.df.apply(lambda x: os.path.join("images", str(x.id) + ".jpg"), axis=1)

        # drop rows where id.jpg cannot be found in the images directory
        self.df = self.df.drop([32309, 40000, 36381, 16194, 6695])

        # map articleType as number
        mapper = {}
        for i, cat in enumerate(list(self.df.articleType.unique())):
            mapper[cat] = i
        self.df['targets'] = self.df.articleType.map(mapper)

    # Get the length
    def __len__(self):
        return len(self.df)

    def load_data(self):

        self.x_train, self.y_train = self.get_i_items(self.df, 800, train=True)
        self.x_test, self.y_test = self.get_i_items(self.df, 200, train=False)

        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def get_i_items(self, df, number_of_items, train):
        # get i items of each condition

        # calculate classes with more than 1000 items
        temp = df.targets.value_counts().sort_values(ascending=False)[:10].index.tolist()
        df_temp = df[df["targets"].isin(temp)]

        x_data = []
        y_data = []

        if train == True:
            for label in temp:

                train_temp = df_temp[df_temp.targets == label]
                train_temp = train_temp[:number_of_items]

                y_data.extend(train_temp["targets"].to_list())

                for element in train_temp.image_path:
                    img = cv2.imread(element)
                    img = cv2.resize(img, (60, 80))
                    img = np.array(img).astype('float32')
                    x_data.append(img)

                # print("Anzahl x_train items bei ", label, " :", len(x_data))
                # print(" ")
        else:
            for label in temp:

                test_temp = df_temp[df_temp.targets == label]
                test_temp = test_temp[800:1000]

                y_data.extend(test_temp["targets"].to_list())

                for element in test_temp.image_path:
                    img = cv2.imread(element)
                    img = cv2.resize(img, (60, 80))
                    img = np.array(img).astype('float32')
                    x_data.append(img)

                # print("Anzahl x_test items bei ", label, " :", len(x_data))
                # print(" ")

        x_data = np.array(x_data)
        # print("input data shape: ",x_data.shape)
        return x_data, y_data

