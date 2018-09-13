import matplotlib.pyplot as plt
from json import load
from data import DATAP, CATS
from pandas import read_csv
from io import StringIO


def plot_allsl_depth(ton):
    f = open(DATAP + '/olangdict.json', 'r', encoding="UTF8")
    langdict = load(f)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in CATS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(CATS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in CATS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + CATS,
                  dtype=dtypes)
    print(df)
    df.plot(x="depth", y=CATS, kind="bar", ax=ax, logy=False, width=0.8, color=["red", "green", "blue"])

    ax.set_title('Articles at Depth')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def plot_seed_depth(ton):
    f = open(DATAP + '/olangdict.json', 'r', encoding="UTF8")
    langdict = load(f)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in CATS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))
                                                 and langdict[cl]["Seed"] == 1])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(CATS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in CATS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + CATS,
                  dtype=dtypes)
    print(df)
    df.plot(x="depth", y=CATS, kind="bar", ax=ax, logy=False, width=0.8, color=["red", "green", "blue"])

    ax.set_title('Articles at Depth')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def plot_negative_seed_depth(ton):
    f = open(DATAP + '/langdict.json', 'r', encoding="UTF8")
    langdict = load(f)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    depthlist = []
    for c in CATS:
        depthlist.append(list(map(lambda d: len([cl for cl in langdict
                                                 if
                                                 ((c + "Depth" in langdict[cl]) and (langdict[cl][c + "Depth"] == d))
                                                 and langdict[cl]["negativeSeed"] == 1])
                                  , range(ton))))

    csvtext = ""
    for n in range(ton):
        csvtext += str(n)
        for i in range(len(CATS)):
            csvtext += ", " + str(depthlist[i][n])
        csvtext += "\n"

    dtypes = dict()
    dtypes["depth"] = int
    for c in CATS:
        dtypes[c] = int

    df = read_csv(StringIO(csvtext), delimiter=',', names=["depth"] + CATS,
                  dtype=dtypes)
    print(df)
    df.plot(x="depth", y=CATS, kind="bar", ax=ax, logy=False, width=0.8, color=["red", "green", "blue"])

    ax.set_title('Negative Seed Distribution')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


if __name__ == "__main__":
    plot_allsl_depth(9)
