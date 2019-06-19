from builtins import range

from cluster.util import plot_3d, plot_2d, create_cluster, cluster_scores, save_classes
from data import DATAP, load_articledict, load_seedlist, save_articledict,start_time, stop_time
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import laplacian_kernel
import matplotlib.pyplot as plt
from data import create_dataframe


# %matplotlib inline

CLUSTERS_ALGORITHMS = {
    "kmeans": lambda n: KMeans(n_clusters=n),
    "hierarchical": lambda n: AgglomerativeClustering(n_clusters=n),
    "gaussian": lambda n: GaussianMixture(n_components=n),

}

FEATURE_NAMES = [
    "InternalWikiLinks",
    "DbpediaInfoboxTemplate",
    "Lemmas",
    "URL_Braces_Words",
    "COPHypernym",
    "Wikipedia_Lists"
]
N_CLUSTERS = 5
N_MOST_IMPORTANT_FEATURES = 1000

if __name__ == "__main__":
    ad = load_seedlist()
    print("Got %s samples without stubs in seed." % str(len(ad)))

    configs = {
        # "single_InternalWikiLinks_3": ["InternalWikiLinks"],
        # "single_DbpediaInfoboxTemplate": ["DbpediaInfoboxTemplate"],
        # "single_Lemmas": ["Lemmas"],
        # "single_URL_Braces_Words": ["URL_Braces_Words"],
        # "single_COPHypernym": ["COPHypernym"],
        # "single_Wikipedia_Lists": ["Wikipedia_Lists"],
        # "multi_Lemmas": ["Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        "multi_InternalLinks_5": ["InternalWikiLinks", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_simple_3": ["URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"]
    }
    N = 5

    cluster_func = lambda n: SpectralClustering(
                                    n_clusters=n,
                                    affinity="precomputed",
                                    random_state=42
                                )

    for (name, config) in configs.items():
        print("Clustering %s" % name)
        df = create_dataframe(ad, indicators=config)

        clustered = create_cluster(df, cluster_func(N), affinity_matrix=laplacian_kernel(df))
        plot_3d(clustered, name)
        save_classes(clustered, name=name)

        # scores = cluster_scores(df, cluster_instantiator=cluster_func, max_n=10)
        # ss = ss.append(pd.Series([name] + [s["silhouette"] for s in scores]), ignore_index=True)
        # dbs = dbs.append(pd.Series([name] + [s["calinski_harabasz"] for s in scores]), ignore_index=True)

    #
    # ss.to_csv(DATAP + "/cluster/silhouettes.csv")
    # dbs.to_csv(DATAP + "/cluster/calinski_harabasz_score.csv")

    # ss = pd.read_csv(DATAP + "/cluster/silhouettes.csv")
    # dbs = pd.read_csv(DATAP + "/cluster/calinski_harabasz_score.csv")
    # x = range(2, 10, 1)
    # fig = plt.figure()
    # ax = plt.subplot(111)
    #
    # plt.xlabel("No. of clusters")
    # plt.ylabel("Avg. silhouette coeff.")
    # plt.set_cmap(plt.get_cmap("Spectral"))
    #
    # for i in range(ss.shape[0]):
    #     label = ss.iloc[i, 0]
    #
    #     y = ss.iloc[i, 1:]
    #     ax.plot(x, y, label=label)
    #
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])
    #
    # plt.show()

    # print(dbs)
    #
    # ################################
    # x = range(2, 10, 1)
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # plt.xlabel("No. of clusters")
    # plt.ylabel("Avg. calinski_harabasz_score")
    # plt.set_cmap(plt.get_cmap("Spectral"))
    #
    # for i in range(dbs.shape[0]):
    #     label = dbs.iloc[i, 0]
    #
    #     if label == "single_DbpediaInfoboxTemplate":
    #         continue
    #
    #     y = dbs.iloc[i, 1:]
    #     ax.plot(x, y, label=label)
    #
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])
    #
    # ax.legend(loc="upper right", fontsize="xx-small")
    # plt.show()






