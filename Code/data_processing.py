import pandas as pd
from urllib.request import urlretrieve
# import labelencoder
from sklearn.preprocessing import LabelEncoder


def load_data(download=True):
    # download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")
        print("Downloaded to car.csv")

    # use pandas to view the data structure
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("../data/car.csv", header=None, names=col_names)
    return data


def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data, prefix=data.columns)


def convert2label(data):
    # instantiate labelencoder object
    le = LabelEncoder()
    # apply le on categorical feature columns
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data[col_names] = data[col_names].apply(lambda col: le.fit_transform(col))
    return data


if __name__ == "__main__":
    data = load_data(download=False)
    print(data.head())
    print("\nNum of data: ", len(data), "\n")  # 1728
    # view data values
    for name in data.keys():
        print(name, pd.unique(data[name]))

    new_data = convert2onehot(data)
    print("\n", new_data.head(2))
    new_data.to_csv("../data/car_onehot.csv", index=False)

    new_data2 = convert2label(data)
    print(new_data2.tail(10))
    new_data2.to_csv("../data/car_labelencoder.csv", index=False)