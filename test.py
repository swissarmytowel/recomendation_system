from datetime import datetime

import numpy as np
from skimage import io, color

from searchroutines import queries, engine
from searchutils import detection, indices

SHOULD_RUN_SEARCH = True

LOAD_LABELS_FROM_DIRECTORY = True
LOAD_FEATURES_FROM_FILE = True

SAMPLE_IMAGE = 75

INDEXED_DATASET_PATH = 'data/indices.pkl'
IMAGES_DIRECTORY = "resources/test_images"
LABELS_DIRECTORY = "resources/segmented_labels"

if __name__ == '__main__':
    print("===========================================================")
    print("======================= START =============================")

    start_time = datetime.now()
    indexed_dataset = queries.index_data_set(directory_name=IMAGES_DIRECTORY,
                                             label_directory=LABELS_DIRECTORY,
                                             pickle_file_name=INDEXED_DATASET_PATH,
                                             indices_are_computed=LOAD_FEATURES_FROM_FILE,
                                             labels_are_computed=LOAD_LABELS_FROM_DIRECTORY)
    if SHOULD_RUN_SEARCH:
        image_name = IMAGES_DIRECTORY + "/" + str(SAMPLE_IMAGE) + ".jpg"
        labels_name = LABELS_DIRECTORY + "/" + str(SAMPLE_IMAGE) + ".txt"
        image = io.imread(image_name)
        try:
            labels = np.loadtxt(labels_name).reshape(image.shape[0], image.shape[1]).astype(np.int)
        except FileNotFoundError as e:
            labels = detection.superpixel_extraction(image)
            np.savetxt(labels_name, X=labels)
        reference_index = detection.index_image(image=image, labels=labels, image_name=image_name)
        search_engine = engine.Engine(sample_image=image, data_set=indexed_dataset, sample_index=reference_index,
                                      relevant_count=20, sensitivity=71)
        search_engine.run()
        search_engine.display_results(mode='write')

    time_elapsed = datetime.now() - start_time

    print("\n======================== END ==============================")
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print("===========================================================")

































    #
    # if RECOMPUTE_LABELS_FROM_TEST_IMAGES:
    #        queries.precompute_labels(image_dir=IMAGES_DIRECTORY, labels_dir=LABELS_DIRECTORY)
    #    else:
    #        image_name = IMAGES_DIRECTORY + "/" + str(SAMPLE_IMAGE) + ".jpg"
    #        labels_name = LABELS_DIRECTORY + "/" + str(SAMPLE_IMAGE) + ".txt"
    #        image = io.imread(image_name)
    #        try:
    #            labels = np.loadtxt(labels_name).reshape(image.shape[0], image.shape[1]).astype(np.int)
    #        except:
    #            labels = detection.superpixel_extraction(image)
    #            np.savetxt(labels_name, X=labels)
    #
    #        features = detection.extract_features(image=image, image_name=image_name, labels=labels)
    #        features_set = list()
    #
    #        if LOAD_FEATURES_FROM_FILE:
    #            features_dict = queries.read_from_file(INDEXED_DATASET_PATH)
    #            tmp_features = dict((IMAGES_DIRECTORY + "/" + str(key) + ".jpg",
    #                                 features_dict[IMAGES_DIRECTORY + "/" + str(key) + ".jpg"]) for key in TEST_KEYS)
    #            features_set = list(indices.ImageIndex(features_pair=i) for i in features_dict.items())
    #            print('Features loaded')
    #        else:
    #            features_set = queries.init_query_from_directory(directory_name=IMAGES_DIRECTORY,
    #                                                             label_directory=LABELS_DIRECTORY)
    #            queries.dump_to_file(features_set, INDEXED_DATASET_PATH)
    #
    #        search_engine = engine.Engine(sample_image=image, data_set=features_set, sample_features=features,
    #                                      relevant_count=20)
    #        search_engine.run()
    #        time_elapsed = datetime.now() - start_time
    #
    #        print("\n======================== END ==============================")
    #        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #        print("===========================================================")
    #
    #        search_engine.display_results()
