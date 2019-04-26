from docutils.nodes import label, General

from data import DATAP, load_articledict, save_articledict,start_time, stop_time
from data.explore.feature_freq import analyze_feature_frequency
from pandas import DataFrame, Series
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

FEATURE_NAMES = ["DbpediaInfoboxTemplate", "Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"]  # Lemmas,
N_CLUSTERS = 2

def build_feature_vector(articles):
    features_freqs = analyze_feature_frequency(articles, FEATURE_NAMES)
    # print(features_freqs)
    return features_freqs.keys()


def build_feature_matrix(features, articles):
    feature_matrix = {f: [] for f in features}
    for a in articles.items():
        # print(a)
        feature_freq = analyze_feature_frequency({a[0]: a[1]}, FEATURE_NAMES)
        for key in feature_matrix:
            feature_matrix[key].append(1 if key in feature_freq else 0)
    return feature_matrix


def create_dataframe(articles):
    print("Creating data frame...")
    features = build_feature_vector(articles)
    feature_matrix = build_feature_matrix(features, articles)
    feature_matrix["Name"] = [a for a in articles]

    df = DataFrame(feature_matrix, columns=["Name"] + list(features))
    df = df.set_index("Name")
    return df


def plot_reduced(df, n_clusters, decompositer = PCA):
    reduced = df
    print("Plotting two-dimensional dataframe reduction...")
    # reduced = DataFrame(decompositer(n_components=2).fit_transform(df.iloc[:, 0:-1]))
    # reduced = reduced.assign(Class=Series(df.loc[:, "Class"].values, index=reduced.index))
    # print(reduced)
    for n in range(n_clusters):
        plt.scatter(reduced[reduced["Class"] == n].iloc[:, 0], reduced[reduced["Class"] == n].iloc[:, 1], cmap=plt.get_cmap("Spectral"), label= "Class %s" % n)
    plt.legend()
    plt.show()


def analyze_clusters(ad, df, n_clusters):
    print("Analyzing clusters...")
    for n in range(n_clusters):
        analyze_cluster(ad, df[df["Class"] == n])


def analyze_cluster(ad, cluster):
    article_names = cluster.index.values

    print("Relevant features: %s" % str(cluster.columns.values))

    articles = dict([a for a in ad.items() if a[0] in article_names])
    print("Feature Frequency for cluster %s" % cluster.loc[:, "Class"][0])
    print(dict(sorted(analyze_feature_frequency(articles, F_SetNames=FEATURE_NAMES).items(), key=lambda v: v[1], reverse=True)))


if __name__ == "__main__":
    ad = load_articledict()
    seed = dict([a for a in ad.items() if a[1]["Seed"] == 1 and a[1]["IsStub"] == 0])
    print("Got %s samples without stubs in seed." % str(len(seed)))
    # seed = dict(list(seed.items())[0:2])
    # print(seed)
    df = create_dataframe(seed)

    corr = df.corr()

    f, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('Correlation Heatmap', fontsize=14)

    plt.show()

    # print("Selecting features...")
    # k_select = SelectKBest(score_func=chi2, k=100)
    # df = k_select.fit(df)
    # print("Applying FastICA to seed...")
    # pca = PCA(n_components=50)
    # df = DataFrame(pca.fit_transform(df), index=df.index)
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


