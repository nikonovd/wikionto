from builtins import range

from cluster.util import plot_3d, plot_2d, create_cluster, cluster_scores, save_classes, best_pca
from data import DATAP, load_articledict, load_seedlist, save_articledict,start_time, stop_time
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from data import create_dataframe
import numpy as np
from sklearn.decomposition import PCA
from pprint import pprint

def plot_score(ss, max_n, title):
    x = range(2, max_n, 1)
    # fig = plt.figure()
    ax = plt.subplot(111)

    plt.xlabel("# clusters")
    plt.ylabel(title)
    plt.set_cmap(plt.get_cmap("Spectral"))

    # ax.set_xticks([2, 3, 4, 5] + [10, 15, 20, 25, 30])

    for i in range(ss.shape[0]):
        label = ss.iloc[i, 0]

        y = ss.iloc[i, 1:]
        ax.plot(x, y, label=label)

        max_x = np.argmax(y) + 1
        max_y = np.max(y)

        print(max_x)
        print(max_y)

        ax.plot([max_x], [max_y], color="red", marker="o")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])

    plt.grid(True, linestyle="--")
    plt.legend()
    plt.show()


def eval_manual(clustered, apps, libs, frameworks):

    scores = {
        "apps": [],
        "libs": [],
        "frameworks": []
    }
    print("a size", len(apps))
    print("l size", len(libs))
    print("f size", len(frameworks))


    clustered = clustered.assign(Name=pd.Series(clustered.index, index=clustered.index))

    for clazz in sorted(set(clustered.loc[:, "Class"].values)):
            c_i = clustered[clustered["Class"] == clazz]
            print("Class", clazz)
            print(c_i.shape)
            app_score = np.sum(np.isin(c_i["Name"].values, [a[0] for a in apps])) / len(apps)
            framework_score = np.sum(np.isin(c_i["Name"].values, [a[0] for a in frameworks])) / len(frameworks)
            lib_score = np.sum(np.isin(c_i["Name"].values, [a[0] for a in libs])) / len(libs)

            print("a", np.sum(np.isin(c_i["Name"].values, [a[0] for a in apps])))
            print("f", np.sum(np.isin(c_i["Name"].values, [a[0] for a in frameworks])))
            print("l", np.sum(np.isin(c_i["Name"].values, [a[0] for a in libs])))

            scores["apps"].append(app_score)
            scores["libs"].append(lib_score)
            scores["frameworks"].append(framework_score)

    pprint(scores)
    print("+++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    ad = load_seedlist()

    frameworks = [a for a in ad.items() if "framework" in a[1]["COPHypernym"]]
    apps = [a for a in ad.items() if "application" in a[1]["COPHypernym"]]
     # url_frameworks = [a for a in ad.items() if "framework" in a[1]["URL_Braces_Words"] and "framework" not in a[1]["COPHypernym"]]
    libs = [a for a in ad.items() if "library" in a[1]["COPHypernym"]]

    # seed = dict(frameworks + apps + libs)
    seed = ad
    print("Got %s samples for clustering" % str(len(seed)))

    configs = {
        # "single_InternalWikiLinks": ["InternalWikiLinks"],
        # "single_DbpediaInfoboxTemplate": ["DbpediaInfoboxTemplate"],
        # "single_Lemmas": ["Lemmas"],
        # "single_URL_Braces_Words": ["URL_Braces_Words"],
        # "single_COPHypernym": ["COPHypernym"],
        # "single_Wikipedia_Lists": ["Wikipedia_Lists"],
        # "multi_Lemmas": ["Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_Lemmas_pca": ["Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        "multi_InternalLinks": ["InternalWikiLinks", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_InternalLinks_pca": ["InternalWikiLinks", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_simple": ["URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_simple_pca": ["URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        "multi_no_hypernym": ["DbpediaInfoboxTemplate", "InternalWikiLinks", "Lemmas", "URL_Braces_Words", "Wikipedia_Lists"],
        "multi_no_hypernym_pca": ["DbpediaInfoboxTemplate", "InternalWikiLinks", "Lemmas", "URL_Braces_Words", "Wikipedia_Lists"],
        # "multi_all":  ["DbpediaInfoboxTemplate", "InternalWikiLinks", "Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
        # "multi_all_pca": ["DbpediaInfoboxTemplate", "InternalWikiLinks", "Lemmas", "URL_Braces_Words", "COPHypernym", "Wikipedia_Lists"],
    }
    # N = 6

    best_N = {
        "multi_InternalLinks": 6,
        "multi_no_hypernym": 4,
        "multi_no_hypernym_pca": 5
    }

    # max_n = 10
    cluster_func = lambda n: AgglomerativeClustering(n_clusters=n)

    ss = pd.DataFrame()
    dbs = pd.DataFrame()
    for (name, config) in configs.items():
        print("Clustering %s with n=%i" % (name, best_N[name]))
        df = create_dataframe(seed, indicators=config)

        # vt = VarianceThreshold(threshold=0.05)
        # vt = vt.fit(df)

        # df = df.iloc[:, vt.get_support(indices=True)]
        if "_pca" in name:
            print("Performing pca...")
            df = pd.DataFrame(best_pca(df), index=df.index)

        N = best_N[name]

        c_name = name + "_" + str(N)

        # print("Calculating laplacian matrix...")
        # X = laplacian_kernel(df)

        clustered = create_cluster(df, cluster_func(N))
        # clustered = df.copy(deep=True)
        # clustered = pd.DataFrame(best_pca(df), index=df.index).assign(Class=pd.Series(np.zeros(df.shape[0], dtype=np.int), index=df.index))
        eval_manual(clustered, apps=apps, libs=libs, frameworks=frameworks)

        # plot_2d(clustered, c_name)
        # save_classes(clustered, name=c_name)

        # scores = cluster_scores(df, cluster_instantiator=cluster_func, max_n=max_n, kernel=None)
        # ss = ss.append(pd.Series([name] + [s["silhouette"] for s in scores]), ignore_index=True)
        # dbs = dbs.append(pd.Series([name] + [s["calinski_harabasz"] for s in scores]), ignore_index=True)

    # ss.to_csv(DATAP + "/cluster/silhouettes_simpler.csv")
    # dbs.to_csv(DATAP + "/cluster/calinski_harabasz_score.csv")
    #
    # plot_score(ss, max_n, "Silhouette coeff.")
    # plot_score(dbs, max_n, "Calinski Harabasz")

    # ss = pd.read_csv(DATAP + "/cluster/silhouettes.csv", index_col=0)
    #
    # ss = ss.iloc[0:6, :]
    #
    # # dbs = pd.read_csv(DATAP + "/cluster/calinski_harabasz_score.csv")

    #
    # # print(dbs)
    # #
    # # ################################
    # x = range(2, max_n, 1)
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # plt.xlabel("# clusters")
    # plt.ylabel("Calinski_harabasz_score")
    # plt.set_cmap(plt.get_cmap("Spectral"))
    #
    # for i in range(dbs.shape[0]):
    #     label = dbs.iloc[i, 0]
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






