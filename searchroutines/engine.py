import os
import numpy as np
from skimage import io, color
from searchutils import detection

name = '75_cmc_21'


class Engine(object):
    def __init__(self, sample_image=None, data_set=None, sample_index=None, relevant_count=10, sensitivity=None):
        self.__data_set = dict()
        self.__sample_index = np.asarray(list())
        self.__sample_image = np.asarray(list())
        self.results = dict()
        self.__relevant_images_count = relevant_count
        self.sensitivity = 101.0 - sensitivity if sensitivity is not None else 20.0
        print(self.sensitivity)
        if data_set is not None:
            self.__data_set = data_set
        if sample_image is not None:
            self.__sample_image = sample_image
            if sample_index is not None:
                self.__sample_index = sample_index
            else:
                self.__sample_index = detection.index_image(self.__sample_image)

    def display_results(self, mode='show'):
        if mode == 'show':
            print("displaying top {} matches..".format(self.__relevant_images_count))
            for counter, result in enumerate(
                    sorted(self.results.items(), key=lambda element: element[1], reverse=False)):
                if counter == self.__relevant_images_count:
                    break
                print(result[1])
                io.imshow_collection([self.__sample_image, io.imread(result[0])])
                io.show()
        else:
            os.mkdir('D:/UGWork/res_slic_exp_new/' + name)
            for counter, result in enumerate(
                    sorted(self.results.items(), key=lambda element: element[1], reverse=False)):
                if counter == self.__relevant_images_count:
                    break
                print(result[0].split('/')[len(result[0].split('/')) - 1], result[1])
                io.imsave(fname='D:/UGWork/res_slic_exp_new/' + name + '/' + str(counter + 1) + '.jpg',
                          arr=io.imread(result[0]))

    def run(self):
        print("\nsearching..")
        for query_entry in self.__data_set:
            dis = self.__sample_index.calculate_distance(query_entry, self.sensitivity)
            self.results.update(
                    {query_entry.image_name: dis})
