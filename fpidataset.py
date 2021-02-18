import os
import pandas as pd
from PIL import Image
import numpy as np

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

        self.x_train = self.get_i_items(self.df, 800, train=True)
        self.x_test = self.get_i_items(self.df, 800, train=False)
        self.y_train = self.get_i_labels(self.df, 800)
        self.y_test = self.get_i_labels(self.df, 800)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_i_items(self, df, i, train):
        # get i items of each condition

        # calculate classes with more than 1000 items
        temp = df.targets.value_counts().sort_values(ascending=False)[:10].index.tolist()
        df_temp = df[df["targets"].isin(temp)]

        # generate new empty dataframe with the columns of the original
        dataframe = df[:0]

        # for each targetclass in temp insert i items in dataframe
        list = []

        if train==True:
            for element in temp:
                # print("Füge Items mit target", element, "ein.")
                dataframe = dataframe.append(df_temp[df_temp.targets == element][:i])
                for x in range(i):
                    img = Image.open(dataframe.image_path[i-1])
                    list.append(img)
                list = np.array(list)
                print("Anzahl x_train itemsbei ", element, " :", len(dataframe))
        else:
            for element in temp:
                dataframe = dataframe.append(df_temp[df_temp.targets == element][i:i + 200])
                print("Anzahl x_test items bei ", element, " :", len(dataframe))

        return dataframe

    def get_i_labels(self, df, i):
        y_train = df.targets[0:i]
        y_test = df.targets[i - 1:i + 200]

        #print("y_train Ausgabe: ", y_train)
        #print("y_test Ausgabe: ", y_test)

        return y_train, y_test


