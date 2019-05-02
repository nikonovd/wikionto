from docutils.nodes import label, General

from data import DATAP, load_articledict, save_articledict,start_time, stop_time
from data.explore.feature_freq import analyze_feature_frequency
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from skfeature.function.similarity_based import SPEC, lap_score

# %matplotlib inline

FEATURE_NAMES = [
    "DbpediaInfoboxTemplate",
    "Lemmas",
    "URL_Braces_Words",
    "COPHypernym",
    "Wikipedia_Lists"
]
N_CLUSTERS = 5
N_MOST_IMPORTANT_FEATURES = 1000

def build_feature_vector(articles):
    features_freqs = analyze_feature_frequency(articles, F_SetNames=FEATURE_NAMES)
    # print(features_freqs)
    return features_freqs.keys()


def build_feature_matrix(features, articles):
    feature_matrix = {f: [] for f in features}
    for a in articles.items():
        # print(a)
        feature_freq = analyze_feature_frequency({a[0]: a[1]}, F_SetNames=FEATURE_NAMES)
        for key in feature_matrix:
            feature_matrix[key].append(1 if key in feature_freq else 0)
    return feature_matrix


def create_dataframe(articles):
    print("Creating data frame...")
    features = build_feature_vector(articles)
    feature_matrix = build_feature_matrix(features, articles)
    feature_matrix["Name"] = [a for a in articles]

    df = pd.DataFrame(feature_matrix, columns=["Name"] + list(features))
    df = df.set_index("Name")
    return df


def plot_2d(df, n_clusters):
    # reduced = df
    print("Plotting 2D dataframe reduction...")
    reduced = pd.DataFrame(PCA(n_components=2).fit_transform(df.iloc[:, 0:-1]))
    reduced = reduced.assign(Class=pd.Series(df.loc[:, "Class"].values, index=reduced.index))

    for n in range(n_clusters):
        plt.scatter(reduced[reduced["Class"] == n].iloc[:, 0], reduced[reduced["Class"] == n].iloc[:, 1],  cmap=plt.get_cmap("Spectral"), label="Class %s" % n)
    plt.legend()
    plt.show()


def plot_3d(df, n_clusters):
    # reduced = df
    print("Plotting 3D dataframe reduction...")
    reduced = pd.DataFrame(PCA(n_components=3).fit_transform(df.iloc[:, 0:-1]))
    reduced = reduced.assign(Class=pd.Series(df.loc[:, "Class"].values, index=reduced.index))

    fig = plt.figure()
    ax = Axes3D(fig)

    for n in range(n_clusters):
        ax.scatter(reduced[reduced["Class"] == n].iloc[:, 0], reduced[reduced["Class"] == n].iloc[:, 1], reduced[reduced["Class"] == n].iloc[:, 2], cmap=plt.get_cmap("Spectral"), label="Class %s" % n)
    plt.legend()
    plt.show()

def explore_silhouette(df, n_clusters):
    silhouettes = []
    for n in range(2, n_clusters, 1):
        clustered = create_cluster(reduced_df, n)
        silhouette = cluster_scores(clustered)
        silhouettes.append((n, silhouette))

    xs = [s[0] for s in silhouettes]
    ys = [s[1] for s in silhouettes]
    plt.plot(xs, ys, '-bX', markevery=True)
    plt.show()

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
        print(dict(sorted(analyze_feature_frequency(articles, relevant_features=relevant_features).items(), key=lambda v: v[1], reverse=True)))


def calc_feature_importance(data):
    print("Calculating feature importance matrix...")
    X = SPEC.spec(data.values, style=0)
    # X = SPEC.feature_ranking(X, style=0)
    d_spec = pd.DataFrame([X], columns=data.columns.values, index=["SPEC"])
    d_spec = d_spec.sort_values("SPEC", axis=1, ascending=False)
    d_spec.to_csv(DATAP + "/cluster/seed_SPEC.csv")

    return d_spec


def load_feature_importance_matrix():
    print("Loading feature importance matrix...")
    matrix = pd.read_csv(DATAP + "/cluster/seed_SPEC.csv")
    dropped_columns = [c for c in matrix.columns.values[1:] if c.split("::")[0] not in FEATURE_NAMES]
    matrix = matrix.drop(labels=dropped_columns, axis=1)

    return matrix


def slice_n_relevant_features(data, feature_importance, n):
    print("Selecting the %s most relevant features..." % str(n))
    print(data.shape)
    print(feature_importance.shape)
    relevant_columns = feature_importance.iloc[0, 0:n+1].index.values[1:]
    return data.loc[:, relevant_columns]


def create_cluster(data, n):
    X = data
    y = pd.Series(AgglomerativeClustering(n_clusters=n).fit_predict(X), index=X.index)

    data = data.assign(Class=y, index=data.index)
    return data


def cluster_scores(data):
    silhouette = silhouette_score(data.iloc[:, 1:-1], labels=data["Class"].values)
    return silhouette


if __name__ == "__main__":
    ad = load_articledict()
    seed = dict([a for a in ad.items() if a[1]["Seed"] == 1 and a[1]["IsStub"] == 0])
    print("Got %s samples without stubs in seed." % str(len(seed)))
    # seed = dict(list(seed.items())[0:2])
    # print(seed)
    df = create_dataframe(seed)

    # calc_feature_importance(df)
    feature_importance = load_feature_importance_matrix()
    # feature_importance = calc_feature_importance(df)

    reduced_df = slice_n_relevant_features(df, feature_importance, n=N_MOST_IMPORTANT_FEATURES)

    # print(reduced_df.shape)

    explore_silhouette(reduced_df, n_clusters=30)




    # plot_2d(clustered, N_CLUSTERS)
    # analyze_clusters(ad, clustered, N_CLUSTERS, relevant_features=[c for c in feature_importance.columns.values[1:] if "COPHypernym" in c])



    # clustered["Class"].to_csv(DATAP + "/cluster/clusters.csv")
    #
    # print("Creating cluster...")
    # ac = GaussianMixture(n_components=N_CLUSTERS)
    # # ac.fit(df)
    # # y = ac.labels_
    #
    # df = df.assign(Class=Series(ac.fit_predict(df), index=df.index))
    # # print(y)
    # plot_reduced(df, N_CLUSTERS)
    #
    # analyze_clusters(ad, df, N_CLUSTERS)


