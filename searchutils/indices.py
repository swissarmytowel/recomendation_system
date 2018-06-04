import math
import numpy as np
from collections.abc import Sequence
from skimage import color, measure, io, future
from searchutils import detection
from sklearn.cluster import KMeans
from multiprocessing import Pool
from scipy import optimize


class IndexFeature(object):
    def __init__(self, image=None, labels=None, label=None, feature=None):
        self.mean_color = list()
        if feature is not None:
            self.mean_color = np.asarray(feature[0])
        else:
            self.label = label
            self.extract(image, labels, label)

    def extract(self, image, labels, label):
        self.mean_color = np.mean(image[labels == label], axis=0)

    def __eq__(self, other):
        return np.equal(self.mean_color, other.mean_color).all()

    def calculate_distance(self, other, sensitivity):
        delta_e = color.deltaE_ciede2000(self.mean_color, other.mean_color)
        return 1 - math.exp(-(delta_e ** 2)/10)


class ImageIndex(Sequence):
    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)

    def calculate_distance(self, other, sensitivity):
        distance_matrix = np.array([
            np.array([sample_feature.calculate_distance(reference_feature, sensitivity) for
                      reference_feature in other.features]) for sample_feature in self.features
        ])
        h, w = len(self.features), len(other.features)
        if h != w:
            if h > w:
                for i in range(h - w):
                    distance_matrix = np.c_[distance_matrix, np.ones(h)]
            else:
                for i in range(w - h):
                    distance_matrix = np.r_[distance_matrix, [np.ones(w)]]
        i, j = 0, 0
        try:
            i, j = optimize.linear_sum_assignment(distance_matrix)
        except ValueError:
            print(self.image_name)
        return distance_matrix[i, j].mean()

    def get_features(self):
        return [(i.mean_color, i.std, i.relative_area) for i in self.features]

    def __init__(self, image=None, labels=None, image_name=None, features_pair=None):
        self.errors = np.ndarray(0)
        if features_pair is not None:
            self.labels = None
            self.features = [IndexFeature(feature=f) for f in features_pair[1]]
            self.image_name = features_pair[0]
        else:
            self.labels = labels
            self.image_name = image_name
            self.features = np.array(
                    [IndexFeature(image=image, labels=labels, label=label) for label in
                     np.unique(labels) if
                     label != 0])
        super().__init__()























        # def filter_features(self):
        #     indices_to_merge = {}
        #     used = []
        #     for i, feature_1 in enumerate(self.features):
        #         local_merging_indices = []
        #         for j, feature_2 in enumerate(self.features):
        #             if feature_1 == feature_2 or j in used:
        #                 continue
        #             if color.deltaE_ciede2000(feature_1.mean_color, feature_2.mean_color) <= 3.0:
        #                 local_merging_indices.append(j)
        #                 used.append(j)
        #         indices_to_merge.update({i: local_merging_indices})
        #         if len(local_merging_indices) > 0:
        #             used.append(i)
        #     means = {}
        #     a = []
        #     c = len(self.features)
        #     for i in list(indices_to_merge.values()):
        #         if len(i) > 0:
        #             for j in i:
        #                 a.append(j)
        #     ind = [i for i in range(len(self.features)) if i not in a]
        #
        #     for i in ind:
        #         means.update({i: self.features[i].mean_color})
        #     for i in indices_to_merge:
        #         if len(indices_to_merge[i]) > 0:
        #             means.update(
        #                     {i: np.mean([np.asarray(self.features[j].mean_color) for j in indices_to_merge[i]], axis=0)})
        #
        #     self.features = [IndexFeature(feature=np.mean([means[i], self.features[i].mean_color], axis=0)) for i in means]
        #     print(c, len(self.features))
