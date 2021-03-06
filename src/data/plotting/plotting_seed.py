import matplotlib.pyplot as plt
from data import ROOTS, load_articledict
from pandas import read_csv
from io import StringIO


def plot_allsl_depth(ton):
    langdict = load_articledict()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in ROOTS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(ROOTS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in ROOTS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + ROOTS,
                  dtype=dtypes)
    print(df)
    df.plot(x="depth", y=ROOTS, kind="bar", ax=ax, logy=False, width=0.8, color=["red", "green", "blue"])

    ax.set_title('Articles at Depth')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def plot_seed_depth(ton):
    langdict = load_articledict()
    #fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in ROOTS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))
                                                 and langdict[cl]["Seed"] == 1])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(ROOTS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in ROOTS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + ROOTS, header=None,
                  dtype=dtypes, index_col=0)
    #df = df.set_index('depth')
    print(df.T.to_latex())
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 42}
    plt.rc('font', **font)

    ax = df.plot(y=ROOTS, kind="bar", logy=True, width=0.9, color=["red", "green", "blue"])

    ax.set_title('Seed articles per depth.')

    #for p in ax.patches:
    #    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def plot_negative_seed_depth(ton):
    langdict = load_articledict()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in ROOTS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))
                                                 and langdict[cl]["negativeSeed"] == 1])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(ROOTS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in ROOTS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + ROOTS,
                  dtype=dtypes)
    print(df.to_latex)
    df.plot(x="depth", y=ROOTS, kind="bar", ax=ax, logy=False, width=0.8, color=["red", "green", "blue"])

    ax.set_title('Negative Seed Distribution')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


if __name__ == "__main__":
    plot_seed_depth(9)
