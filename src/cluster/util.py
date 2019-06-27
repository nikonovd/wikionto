from sklearn.decomposition import PCA, FastICA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import laplacian_kernel
from data import DATAP
from data.explore.feature_freq import analyze_feature_frequency
import numpy as np
from sklearn.decomposition import PCA

def plot_2d(df, title="", export=False):
    # reduced = df
    print("Plotting 2D dataframe reduction...")
    reduced = pd.DataFrame(PCA(n_components=2, random_state=42).fit_transform(df.loc[:, df.columns != "Class"]))
    reduced = reduced.assign(Class=pd.Series(df.loc[:, "Class"].values, index=reduced.index))

    for n in set(reduced.loc[:, "Class"].values):
        plt.scatter(reduced[reduced["Class"] == n].iloc[:, 0], reduced[reduced["Class"] == n].iloc[:, 1],  cmap=plt.get_cmap("Spectral"), label="Class %s" % n)

    plt.title(title)

    plt.legend()
    plt.show()

    if export:
        plt.savefig(DATAP + "/cluster/" + title + ".png")


def plot_3d(df, title="", export=False):
    # reduced = df
    print("Plotting 3D dataframe reduction...")
    reduced = pd.DataFrame(PCA(n_components=3, random_state=42).fit_transform(df.loc[:, df.columns != "Class"]))
    reduced = reduced.assign(Class=pd.Series(df.loc[:, "Class"].values, index=reduced.index))

    fig = plt.figure()
    ax = Axes3D(fig)

    for n in set(reduced.loc[:, "Class"].values):
        ax.scatter(reduced[reduced["Class"] == n].iloc[:, 0], reduced[reduced["Class"] == n].iloc[:, 1], reduced[reduced["Class"] == n].iloc[:, 2], cmap=plt.get_cmap("Spectral"), label="Class %s" % n)

    plt.title(title)
    plt.legend()
    plt.show()

    if export:
        plt.savefig(DATAP + "/cluster/" + title + ".png")


def create_cluster(data, clusterer, affinity_matrix=None):
    print("Creating clusters...")
    X = data
    if affinity_matrix is not None:
        X = affinity_matrix
    y = pd.Series(clusterer.fit_predict(X), index=data.index)

    clustered = data.copy(deep=True)
    clustered = clustered.assign(Class=y)
    return clustered


def cluster_scores(df, cluster_instantiator, max_n, kernel=laplacian_kernel):
    silhouettes = []

    affinity_matrix = None
    if kernel is not None:
        print("Calculating laplacian matrix")
        affinity_matrix = kernel(df.values)

    for n in range(2, max_n, 1):
        print("Step %s" % str(n))
        c = cluster_instantiator(n)
        clustered = create_cluster(data=df, clusterer=c, affinity_matrix=affinity_matrix)
        scores = __cluster_scores(clustered)
        silhouettes.append(scores)

    return silhouettes
    # xs = [s[0] for s in silhouettes]
    # ys = [s[1] for s in silhouettes]
    # plt.plot(xs, ys, '-bX', markevery=True)
    # plt.show()


def __cluster_scores(data):
    return {
        "silhouette": silhouette_score(data.iloc[:, 1:-1], labels=data.loc[:, "Class"].values),
        "calinski_harabasz": calinski_harabasz_score(data.iloc[:, 1:-1], labels=data.loc[:, "Class"].values)
    }


def save_classes(data, name):
    data["Class"].to_csv(DATAP + "/cluster/%s.csv" % name)


def analyze_clusters(ad, df, n_clusters, relevant_features):
    print("Analyzing clusters...")
    # print("Relevant features: %s" % str(relevant_features))
    for n in range(n_clusters):
        analyze_cluster(ad, df[df["Class"] == n], relevant_features)


def analyze_cluster(ad, cluster, relevant_features=None):
    article_names = cluster.index.values

    articles = dict([a for a in ad.items() if a[0] in article_names])
    print("%s articles clustered in cluster %s" % (len(articles), cluster.loc[:, "Class"][0]))
    print("Feature Frequency for cluster %s" % cluster.loc[:, "Class"][0])
    if relevant_features is not None:
        print(dict(sorted(analyze_feature_frequency(articles, relevant_features=relevant_features).items(),
                          key=lambda v: v[1], reverse=True)))


def best_pca(data):
    pca = PCA().fit(data)
    best_n_components = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.998)

    print("...reducing to %s components." % str(best_n_components[0][0]))

    return PCA(n_components=best_n_components[0][0]).fit_transform(data)
