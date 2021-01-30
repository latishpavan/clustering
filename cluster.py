import random
import warnings
from typing import List

# external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import DataConversionWarning

# suppress warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore', category=DataConversionWarning)


C_MAX = 8
MAX_ITERATIONS = 10


# distance
def pairwise_distance(source, dest):
    source, dest = np.array(source), np.array(dest)
    # distance = np.dot(source, dest) / np.sqrt(np.sum(source ** 2) * np.sum(dest ** 2))
    # return distance
    return np.sqrt(np.sum((dest - source) ** 2, axis=0))


# all vectors in our dataset modelled with CLusterPoint
class ClusterPoint:
    def __init__(self, vector, index: int):
        self.index: int = index
        self.vector = vector.tolist() if isinstance(vector, np.ndarray) else vector
        self.cluster_id = None
        self.is_repr = False

    def assign_cluster(self, cluster_id: int):
        self.cluster_id = cluster_id

    def get_cluster_id(self):
        return self.cluster_id

    def get_vector(self):
        return self.vector

    def get_index(self):
        return self.index

    def __eq__(self, other):
        return self.vector == other.get_vector()

    def __ne__(self, other):
        return not self.__eq__(other)

    def distance_from(self, other):
        return pairwise_distance(self.vector, other.get_vector())

    def is_representative(self) -> bool:
        return self.is_repr

    def __repr__(self):
        return f'(vector={self.vector}, cluster_id={self.cluster_id}), index={self.index}, is_repr={self.is_repr})'


# ClusterRepresentative isA ClusterPoint
# ClusterPoint with some additional functionality
class ClusterRepresentative(ClusterPoint):
    def __init__(self, vector, index: int):
        super(ClusterRepresentative, self).__init__(vector, index)
        self.is_repr = True

    @classmethod
    def from_cluster_point(cls, point: ClusterPoint, cluster_id: int):
        instance = cls(point.get_vector(), point.get_index())
        instance.assign_cluster(cluster_id)
        return instance


# models the cluster in general
class Cluster:
    def __init__(self, cluster_id: int, dataset: List[ClusterPoint], representative: ClusterRepresentative):
        self.dataset = dataset
        self.cluster_id = cluster_id
        self.representative = representative

    # returns all members of the cluster including representative
    def get_members(self) -> List[ClusterPoint]:
        return list(filter(lambda x: x.get_cluster_id() == self.cluster_id, self.dataset))

    def set_representative(self, new_rep: ClusterRepresentative):
        self.representative = new_rep

    def get_representative(self):
        return self.representative

    def set_cluster_id(self, cluster_id: int):
        self.cluster_id = cluster_id

    @staticmethod
    def _get_cluster_mean(members):
        return np.mean([member.get_vector() for member in members], axis=0)

    def recalculate_representative(self) -> bool:
        # Object Median
        members = self.get_members()

        if len(members) > 0:
            cluster_mean = self._get_cluster_mean(members)
            distances_from_mean = [pairwise_distance(member.get_vector(), cluster_mean) for member in members]

            # select the new representative using object median linear time algorithm
            # return True if representative changes, used to stop the iteration
            min_value_index = np.argmin(distances_from_mean)
            if self.representative != members[min_value_index]:
                self.set_representative(ClusterRepresentative.from_cluster_point(members[min_value_index], self.cluster_id))
                return True

        return False

    # Definition 4
    def granularity(self, cluster_scheme):
        members = self.get_members()

        distance_from_rep, membership_value = [], []

        for member in members:
            # ignore cluster representative
            if not member.is_representative():
                distance_from_rep.append(member.distance_from(self.representative) ** 2)
                membership_value.append(cluster_scheme.membership_value(member) ** 2)

        # granularity for a singleton cluster is zero
        if len(members) <= 1:
            return 0

        gs = np.dot(membership_value, distance_from_rep) / np.sum(membership_value)
        return gs

    # Definition 5
    def dissimilarity(self, cluster_scheme):
        clusters = cluster_scheme.get_clusters()
        reps = [cluster.get_representative() for cluster in clusters]
        distance_from_other_reps = [self.representative.distance_from(rep) for ind, rep in enumerate(reps)
                                    if ind != self.cluster_id]
        return np.min(distance_from_other_reps) ** 2


# models the cluster scheme with a set of clusters
class ClusterScheme:
    def __init__(self, clusters: List[Cluster], dataset: List[ClusterPoint]):
        self.clusters = clusters
        self.dataset = dataset

    def get_clusters(self):
        return self.clusters

    # Fuzzy membership function
    def membership_value(self, data_point: ClusterPoint):
        cluster_number = data_point.get_cluster_id()
        cluster_repr = self.clusters[cluster_number].get_representative()
        if cluster_repr.distance_from(data_point) == 0:
            return 1

        for index, cluster in enumerate(self.clusters):
            if cluster_number != index and cluster.get_representative().distance_from(data_point) == 0:
                return 0

        cluster_rep_distances = [cluster.get_representative().distance_from(data_point) for cluster in self.clusters]
        return 1 / (np.sum(1 / np.array(cluster_rep_distances), axis=0) * cluster_rep_distances[cluster_number])

    # Definition 1
    def granularity(self):
        # Take only clusters which are not singleton
        gs_scores = [cluster.granularity(self) for cluster in self.clusters if len(cluster.get_members()) > 1]
        return len(gs_scores), np.mean(gs_scores)

    # Definition 2
    def dissimilarity(self):
        dissimilarity_scores = [cluster.dissimilarity(self) for cluster in self.clusters]
        return np.sum(dissimilarity_scores) / (len(self.clusters) ** 2)

    # Definition 3
    def gd_score(self):
        try:
            k, gr = self.granularity()
            ds = self.dissimilarity()

            return (k / len(self.clusters)) * (ds / gr)
        except ZeroDivisionError:
            print('Zero division encountered')
            return 0

    def get_cluster(self, ind):
        return self.clusters[ind]

    # Procedure 1.1. Recalculate the cluster representatives
    def recalculate_representatives(self):
        current_iteration, changed = 0, False

        while not changed and current_iteration <= MAX_ITERATIONS:
            reps_changed = []
            # Step 1
            for x in self.dataset:
                distances = [cluster.get_representative().distance_from(x) for cluster in self.clusters]
                index = np.argmin(distances, axis=0)
                x.assign_cluster(index)

            # Step 2
            for cluster in self.clusters:
                rep_changed = cluster.recalculate_representative()
                reps_changed.append(rep_changed)

            # Step 3
            changed = all(reps_changed)
            current_iteration += 1

    def remove_cluster(self, cluster_index: int):
        # reduce the cluster index of clusters greater than the cluster index to be removed
        for index in range(cluster_index + 1, len(self.clusters)):
            self.clusters[index].set_cluster_id(index - 1)

        self.clusters.pop(cluster_index)

    def get_worst_cluster(self):
        # remove the cluster with least GD
        gd_scores = []

        for cluster in self.clusters:
            granularity = cluster.granularity(self)

            if granularity != 0:
                gd_scores.append(cluster.dissimilarity(self) / granularity)
            else:
                gd_scores.append(0)

        min_gd_index = np.argmin(gd_scores)
        return min_gd_index


# main clustering algorithm
class MultiStepMaxMinAlgorithm:
    def __init__(self, dataset: List[ClusterPoint], cmax: int):
        self.dataset = dataset
        self.cmax = cmax
        self.copt = None
        self.optimal_scheme = None
        self.gd_values = {}

    # Modified maxmin algorithm
    def modified_maxmin(self, num_clusters: int, initial_index: int) -> ClusterScheme:
        cluster_representatives, visited, i = [], [False] * len(self.dataset), 0

        # append randomly selected representative
        cluster_representatives.append(
            ClusterRepresentative.from_cluster_point(self.dataset[initial_index], i)
        )
        visited[initial_index] = True

        while i < num_clusters:
            iteration_max, iteration_rep_index = float('-inf'), None

            for x in range(0, len(self.dataset)):
                if not visited[x]:
                    distances = [rep.distance_from(self.dataset[x]) for rep in cluster_representatives]
                    min_distance = min(distances)

                    if iteration_max < min_distance:
                        iteration_max = min_distance
                        iteration_rep_index = x

            i += 1
            cluster_representatives.append(
                ClusterRepresentative.from_cluster_point(self.dataset[iteration_rep_index], i))
            visited[iteration_rep_index] = True

        # make clusters
        clusters = [Cluster(representative=rep, cluster_id=index, dataset=self.dataset) for index, rep in
                    enumerate(cluster_representatives)]

        for x in range(0, len(self.dataset)):
            if not visited[x]:
                distances = [rep.distance_from(self.dataset[x]) for rep in cluster_representatives]
                min_rep_index = np.argmin(distances, axis=0)
                self.dataset[x].assign_cluster(min_rep_index)

        return ClusterScheme(clusters=clusters, dataset=self.dataset)

    # Algorithm 2: Multi-step maximum algorithm
    def multistep_maximum(self, num_clusters: int, point_index: int) -> ClusterScheme:
        # Step 1: Initialize variables
        optimal_gd, optimal_cluster = 0, None

        for i in range(num_clusters):
            # Step 2
            cluster_scheme = self.modified_maxmin(num_clusters, point_index)

            # Step 3: recalculate representatives
            cluster_scheme.recalculate_representatives()

            # Step 4: Compare GDs
            current_gd = cluster_scheme.gd_score()
            if current_gd > optimal_gd:
                optimal_gd, optimal_cluster = current_gd, cluster_scheme.get_clusters()

            # Step 5: Update point p
            point_index = cluster_scheme.get_cluster(i).get_representative().get_index()

        return ClusterScheme(clusters=optimal_cluster, dataset=self.dataset)

    # Algorithm 1. Merging algorithm
    @staticmethod
    def merge_clusters(cluster_scheme: ClusterScheme):
        # Step 1
        worst_cluster_index = cluster_scheme.get_worst_cluster()
        cluster_scheme.remove_cluster(worst_cluster_index)
        cluster_scheme.recalculate_representatives()

    # Algorithm 3. Multi-step maxmin and merge algorithm (3M algorithm)
    def fit(self) -> ClusterScheme:
        # Step 1
        c = self.cmax
        initial_point_index = random.randint(0, len(self.dataset) - 1)
        self.optimal_scheme = None
        optima_c, optimal_gd = c, 0

        while c >= 2:
            print(f'Evaluating for c={c} clusters...')

            # Step 1
            current_scheme = self.multistep_maximum(c, initial_point_index)
            current_gd = current_scheme.gd_score()
            self.gd_values[c] = current_gd

            # Step 2
            self.merge_clusters(current_scheme)
            initial_point_index = current_scheme.get_cluster(0).get_representative().get_index()

            if current_gd > optimal_gd:
                optimal_gd, self.optimal_scheme = current_gd, current_scheme
                self.copt = c

            c -= 1

        return self.optimal_scheme

    def validity_scores(self) -> dict:
        return self.gd_values


def plot_validity_scores(validity_scores: dict, plot_name: str):
    print('Plotting validity values...')
    plt.grid(True)
    plt.xlabel('c (number of clusters)')
    plt.ylabel('validity value')
    plt.plot(list(validity_scores.keys()), list(validity_scores.values()), marker='^')
    plt.savefig(plot_name)


def run_clustering(input_norm: np.ndarray):
    input_list = [ClusterPoint(vector=arr, index=index) for index, arr in enumerate(input_norm)]

    instance = MultiStepMaxMinAlgorithm(dataset=input_list, cmax=C_MAX)
    optimal_clusters = instance.fit()
    validity_scores = instance.validity_scores()

    # get representatives
    representatives = [cluster.get_representative() for cluster in optimal_clusters.get_clusters()]

    return validity_scores, representatives


def main():
    # data cleaning
    columns = ['ID', 'Clump thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
               'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    cancer_data = pd.read_csv('./breast-cancer-wisconsin.data', names=columns)
    class_map = {
        2: 'benign',
        4: 'malignant'
    }

    cancer_data['Class'] = cancer_data.Class.map(class_map)
    input_columns = columns[1:-1]
    input_data = cancer_data[input_columns]

    # impute missing values
    input_data['Bare Nuclei'] = input_data['Bare Nuclei'] \
        .replace(to_replace='?', value=np.nan) \
        .fillna(method='bfill') \
        .map(int)

    input_arr = input_data.to_numpy()
    input_arr_norm = MinMaxScaler().fit_transform(input_arr)
    scores, representatives = run_clustering(input_arr_norm)
    print(representatives)

    # plot the scores and save to a file
    plot_validity_scores(scores, 'sample.png')


if __name__ == '__main__':
    main()
