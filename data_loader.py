import pandas as pd
import pprint


def build_ith_dataset(train_attributes_file, train_label_file):
    dataset_i = {5: [], 6: []}

    # Load attributes file and label file
    df_attributes = pd.read_csv(train_attributes_file, header=None)
    df_label = pd.read_csv(train_label_file, header=None)

    if len(df_attributes) != len(df_label):
        raise ValueError('attributes and label file does not match')

    for index, row in df_attributes.iterrows():
        label = df_label.iloc[[index]].values[0][0]
        if label in [5, 6]:
            dataset_i[label].append(list(row.values))
        else:
            raise ValueError('Unknown label encountered')

    return dataset_i


def build_dataset(dataset_path, k=10):

    dataset = []
    for i in range(1, k + 1):
        ith_dataset = build_ith_dataset(dataset_path + '/data' + str(i) + '.csv',
                                        dataset_path + '/labels' + str(i) + '.csv')
        dataset.append(ith_dataset)

    return dataset


if __name__ == '__main__':

    dataset_ith = build_ith_dataset('knn-dataset/data1.csv', 'knn-dataset/labels1.csv')
    pprint.pprint(dataset_ith)

    full_dataset = build_dataset('./knn-dataset')
    print(len(full_dataset))
