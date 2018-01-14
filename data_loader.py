import pandas as pd
from functools import reduce

class DataLoader:
    """
    Class that helps in loading dataset. Assumes attributes and labels are in seperate file
    """

    num_files = 10

    @classmethod
    def load_dataset(cls, attributes_file, labels_file):
        """
        :param attributes_file: file path of attributes
        :param labels_file: file path of correspnding labels
        :return a 2-tuple of dataframes of attributes and labels
        Loads data from single attributes, labels file pair
        """

        # Load attributes file and value file
        df_attributes = pd.read_csv(attributes_file, header=None, dtype=int)
        df_labels = pd.read_csv(labels_file, header=None, dtype=float)

        if len(df_attributes) != len(df_labels):
            raise ValueError('count mismatch in attributes and labels file')

        return df_attributes, df_labels

    @classmethod
    def load_full_dataset(cls, dataset_root_directory):
        """
        :param dataset_root_directory: root directory containing all dataset files
        :return a list of 2-tuples of dataframes of attributes and labels
        Loads data from a collection of files
        """

        full_dataset = []
        for i in range(1, DataLoader.num_files + 1):
            attributes, labels = DataLoader.load_dataset(
                                    dataset_root_directory + '/data' + str(i) + '.csv',
                                    dataset_root_directory + '/labels' + str(i) + '.csv'
                                    )
            full_dataset.append((attributes, labels))
        return full_dataset

    @classmethod
    def load_with_test_data(cls, dataset_root_directory, split_ratio=0.1):

        if not 0 < split_ratio < 1:
            raise ValueError('Split ratio should be in (0,1)')

        # Concatenate all test files data together
        test_files_count = int(split_ratio * DataLoader.num_files)
        attributes_group = []
        labels_group = []
        for i in range(1, test_files_count + 1):
            test_attrs, test_labels = DataLoader.load_dataset(
                                          dataset_root_directory + '/data' + str(i) + '.csv',
                                          dataset_root_directory + '/labels' + str(i) + '.csv')
            attributes_group.append(test_attrs)
            labels_group.append(test_labels)

        test_attrs = pd.concat(attributes_group, ignore_index=True)
        test_labels = pd.concat(labels_group, ignore_index=True)

        # Treat remaining files as training files
        train_dataset = []
        for i in range(test_files_count + 1, DataLoader.num_files + 1):
            attributes, labels = DataLoader.load_dataset(
                dataset_root_directory + '/data' + str(i) + '.csv',
                dataset_root_directory + '/labels' + str(i) + '.csv'
            )
            train_dataset.append((attributes, labels))

        return train_dataset, test_attrs, test_labels


if __name__ == '__main__':

    part_train_attrs, part_train_labels = DataLoader.load_dataset(
                                              'knn-dataset/data1.csv',
                                              'knn-dataset/labels1.csv'
                                          )
    print('='*5 + 'Training data from single file' + '='*5)
    print('Attributes = ', part_train_attrs.shape)
    print('Labels = ', part_train_labels.shape)
    print()

    full_dataset = DataLoader.load_full_dataset('./knn-dataset')
    print('=' * 15 + 'Full dataset' + '=' * 15)
    attr_count = labels_count = 0
    for ds in full_dataset:
        attr_count += ds[0].shape[0]
        labels_count += ds[1].shape[0]
    print('Attributes = ', attr_count)
    print('Labels = ', labels_count)
    print()

    train_dataset, test_attrs, test_labels = DataLoader.load_with_test_data(
                                                            './knn-dataset',
                                                            split_ratio=0.1)
    print('=' * 5 + 'Train-Test split' + '=' * 5)
    attr_count = labels_count = 0
    for ds in train_dataset:
        attr_count += ds[0].shape[0]
        labels_count += ds[1].shape[0]
    print('Train Attributes = ', attr_count)
    print('Train Labels = ', labels_count)
    print('Test Attributes = ', test_attrs.shape)
    print('Test Labels = ', test_labels.shape)