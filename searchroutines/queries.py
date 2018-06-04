import os
import sys
import pickle
from skimage import io, color
import numpy as np
from searchutils import detection, indices


def index_data_set(directory_name: str, label_directory: str, pickle_file_name, indices_are_computed=False,
                   labels_are_computed=False):
    if indices_are_computed:
        print('Indices were already computed. Loading from ' + pickle_file_name)
        index_dictionary = read_from_file(pickle_file_name)
        indexed_dataset = list(indices.ImageIndex(features_pair=i) for i in index_dictionary.items())
    else:
        if not labels_are_computed:
            precompute_labels(directory_name, label_directory)
            print('Labels are pre-computed')
        indexed_dataset = init_query_from_directory(directory_name, label_directory)
        print('Features are calculated')
        dump_to_file(indexed_dataset, pickle_file_name)
        print('Serialized and dumped to file: ' + pickle_file_name)
    return indexed_dataset


def init_query_from_directory(directory_name: str, label_directory: str):
    quantity = len(os.listdir(label_directory))
    counter = 0
    query_features = list()
    sys.stdout.write('\r' + "indexing: " + str(counter) + "/" + str(quantity))
    for name in os.listdir(label_directory):
        counter += 1
        image_name = directory_name + "/" + name.partition('.')[0] + ".jpg"
        try:
            image = io.imread(image_name)
        except (MemoryError, FileNotFoundError):
            image_name = directory_name + "/" + name.partition('.')[0] + '.jpeg'
            try:
                image = io.imread(image_name)
            except (MemoryError, FileNotFoundError):
                print('from imread', image_name)

        labels = np.loadtxt(fname=label_directory + "/" + name).reshape(image.shape[0], image.shape[1]).astype(
                np.int)
        query_features.append(detection.index_image(image=image, labels=labels, image_name=image_name))
        sys.stdout.write('\r' + "indexing: " + str(counter) + "/" + str(quantity))
    return query_features


def dump_to_file(indices, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(dict((i, j) for i, j in [(index.image_name, index.get_features()) for index in indices]), file)


def read_from_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def precompute_labels(image_dir, labels_dir):
    query_names = [name for name in os.listdir(image_dir) if
                   name.endswith(".jpg") or name.endswith(".jpeg")]
    query_names.sort(key=lambda x: int(x.partition(".")[0]))
    size = len(query_names)
    counter = 0
    for name in query_names:
        sys.stdout.write('\r' + "computing labels: " + str(counter) + "/" + str(size))
        query_image = io.imread(image_dir + "/" + name)
        labels = detection.superpixel_extraction(query_image)
        np.savetxt(fname=(labels_dir + "/" + str(name.partition('.')[0]) + ".txt"), X=labels)
        counter += 1
