import json
from mine.util import flatten
from data import DATAP
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from cluster.util import create_cluster, plot_2d, plot_3d, save_classes
from data.explore import  feature_freq
from sklearn.decomposition import PCA


def get_feature_vector(links):
    return sorted(list(set(flatten(links.values()))))


def load_links():
    return json.load(open(DATAP + "/seed_summary_links.json", "r"))


def get_feature_values(links, feature):
    return np.array([feature in links[l] for l in links]).astype(int)


def create_dataframe(links):
    features = get_feature_vector(links)
    articles = list(links.keys())

    matrix = {
        f: get_feature_values(links, f) for f in features
    }
    matrix["name"] = articles
    # print(matrix)

    df = pd.DataFrame(columns=features + ["name"], data=matrix)
    df = df.set_index("name")

    df = preprocess(df)

    return df


def preprocess(df):
    # drop all zero samples
    df = df[(df != 0).any(axis=1)]
    return df


if __name__ == "__main__":
    links = load_links()
    data = create_dataframe(links)

    # data = pd.DataFrame(PCA(n_components=20).fit_transform(data), index=data.index)

    h = SpectralClustering(n_clusters=3, affinity="cosine", random_state=42)
    clustered = create_cluster(data, h)



    # print(clustered[clustered.columns[0:-2]])

    save_classes(clustered)

    plot_3d(clustered, "Spectral n=3")

    grouped = clustered.groupby(by="Class").sum()


    # grouped.to_csv(DATAP+"/cluster/cluster_analysis.csv")

    # pprint(set(flatten(links.values())))
    # pprint(len(get_feature_vector(links)))