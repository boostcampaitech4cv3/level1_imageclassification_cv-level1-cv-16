import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Artists:
    def __init__(self, seed):
        df = pd.read_csv('train.csv')

        self.le = preprocessing.LabelEncoder()
        df['artist'] = self.le.fit_transform(df['artist'].values)

        train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=seed)

        train_df = train_df.sort_values(by=['id'])
        val_df = val_df.sort_values(by=['id'])
        test_df = pd.read_csv('test.csv')

        self.train_img_paths, self.train_labels = self.get_data(train_df)
        self.val_img_paths, self.val_labels = self.get_data(val_df)
        self.test_img_paths = self.get_data(test_df, infer=True)

    def get_data(self, df, infer=False):
        if infer:
            return df['img_path'].values
        return df['img_path'].values, df['artist'].values