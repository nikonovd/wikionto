# my
from data import load_articledict, DATAP
from data.explore.feature_freq import analyze_feature_frequency
from data.eval.random_sampling import get_random_data
# learn
from scipy.sparse import dok_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from yellowbrick.features import RFECV
from classify.dottransformer import transform
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
# util
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from json import dump
from collections import Counter

F_SETNAMES = ["DbpediaInfoboxTemplate", "URL_Braces_Words", "COPHypernym", "Lemmas", "Wikipedia_Lists"]  # Lemmas,


def get_seed(ad):
    S = [a for a in ad if ad[a]["Seed"]]
    y = ['1' for s in S]
    return S, y


def build_f_to_id(FS, ad):
    freq = analyze_feature_frequency(ad, FS)
    fs = [f for f, count in freq.items() if count > 10]

    fids = list(range(len(fs)))
    f_to_id = dict(zip(fs, fids))
    with open(DATAP + '/f_to_id.json', 'w', encoding='utf-8') as f:
        dump(f_to_id, f, indent=1)
    print("Total number of features:" + str(len(f_to_id)))
    return f_to_id, fs


def build_id_to_a(A_p, aid=0):
    matrix_id_to_a = {}
    for a in A_p:
        matrix_id_to_a[aid] = a
        aid += 1
    return matrix_id_to_a


def build_dok_matrix(id_to_a, f_to_id, ad, F_Names):
    flen = len(f_to_id)
    matrix = dok_matrix((len(id_to_a), flen), dtype=numpy.int8)
    for aid, a in id_to_a.items():
        for F_Name in F_Names:
            if F_Name in ad[a]:
                for f in ad[a][F_Name]:
                    if F_Name + "::" + f in ad[a]:
                        matrix[aid, f_to_id[F_Name + "::" + f]] = 1
    return matrix


def get_data_sets(ad, splitnr=1000):
    A_seed, y_seed = get_seed(ad)
    A_random, y_random = get_random_data()

    A_data, y_data = A_seed + A_random, y_seed + y_random

    A_train, y_train = A_data[:splitnr], y_data[:splitnr]
    A_test, y_test = A_data[splitnr:], y_data[splitnr:]

    print("Retrieving data sets:")
    print("Positive in train: " + str(len([y for y in y_train if y == '1'])))
    print("Positive in test: " + str(len([y for y in y_test if y == '1'])))
    (f_to_id, fs) = build_f_to_id(F_SETNAMES, ad)

    return (A_train, y_train), (A_test, y_test), (f_to_id, fs)


def train_decisiontree_with(train_data, k, score_function=chi2, undersam=False, oversam=False, export=False):
    assert k > 0
    print("Training with " + str(k))
    y_train, y_test, f_to_id, id_to_a_train, id_to_a_test = train_data
    dtc = DecisionTreeClassifier(random_state=0)

    print("Build Sparse Matrix")
    X_train = build_dok_matrix(id_to_a_train, f_to_id, ad, F_SETNAMES)
    X_test = build_dok_matrix(id_to_a_test, f_to_id, ad, F_SETNAMES)

    print("Select KBest")
    selector = SelectKBest(score_function, k=k)
    result = selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    fitted_ids = [i for i in result.get_support(indices=True)]

    print("Apply Resampling")
    print(Counter(y_train))
    if undersam and not oversam:
        renn = RepeatedEditedNearestNeighbours()
        X_train, y_train = renn.fit_resample(X_train, y_train)
    if oversam and not undersam:
        smote_nc = SMOTENC(categorical_features=[0, 1], random_state=0)
        X_train, y_train = smote_nc.fit_resample(X_train, y_train)
    if oversam and undersam:
        smote_enn = SMOTEENN(random_state=0)
        X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    print(Counter(y_train))

    print("Train Classifier")
    clf = dtc.fit(X_train, y_train, check_input=True)

    if export:
        export_graphviz(clf, out_file=DATAP + "/temp/lemmatrees2/sltree" + str(k) + ".dot", filled=True)
        transform(fitted_ids, k)

    y_test_predicted = dtc.predict(X_test)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x in range(len(y_test)):
        if y_test[x] == '1' and y_test_predicted[x] == '1':
            tp += 1
        if y_test[x] == '1' and y_test_predicted[x] == '0':
            fn += 1
        if y_test[x] == '0' and y_test_predicted[x] == '0':
            tn += 1
        if y_test[x] == '0' and y_test_predicted[x] == '1':
            fp += 1

    return selector, dtc, {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "k": k, "#Features": len(fitted_ids),
                           "Balanced_Accuracy": balanced_accuracy_score(y_test, y_test_predicted),
                           "F_Measure": f1_score(y_test, y_test_predicted, pos_label='1'),
                           "TPR": recall_score(y_test, y_test_predicted, pos_label='1'),
                           "FPR": fp / (fp + tn),
                           "PPV": precision_score(y_test, y_test_predicted, pos_label='1'),
                           "Self_Accuracy": dtc.score(X_train, y_train)}


def train_decisiontree_exploration(ad, k0=1, kmax=60, kstep=1, splitnr=1000, score_function=chi2, undersam=False,
                                   oversam=False):
    (A_train, y_train), (A_test, y_test), (f_to_id, fs) = get_data_sets(ad, splitnr)
    id_to_a_train = build_id_to_a(A_train)
    id_to_a_test = build_id_to_a(A_test)
    evals = []
    id_to_a_all = build_id_to_a([a for a in ad])
    X_all0 = build_dok_matrix(id_to_a_all, f_to_id, ad, F_SETNAMES)
    for k in range(k0, kmax, kstep):
        train_data = y_train, y_test, f_to_id, id_to_a_train, id_to_a_test
        selector, dtc, eval_dict = train_decisiontree_with(train_data, k, score_function, False, undersam, oversam)
        X_allk = selector.transform(X_all0)
        y_all = dtc.predict(X_allk)
        eval_dict["Positive"] = len([y for y in y_all if y == '1'])
        eval_dict["Negative"] = len([y for y in y_all if y == '0'])
        evals.append(eval_dict)
    return pd.DataFrame(evals)


def crossvalidation_search_decision_tree(ad):
    (A_train, y_train), (A_test, y_test), f_to_id = get_data_sets(ad)
    A = A_train + A_test
    id_to_a_train = build_id_to_a(A)
    y = y_train + y_test
    dtc = DecisionTreeClassifier(random_state=0)
    selector = RFECV(dtc)
    X = build_dok_matrix(id_to_a_train, f_to_id, ad, F_SETNAMES)
    print("built matrix")
    selector = selector.fit(X, y)
    print("finished fit")
    selector.poof()
    # export_graphviz(selector.estimator_, out_file=DATAP + "/temp/trees/sltree_crossval.dot")


def feature_count_positive(ad):
    (A_train, y_train), (A_test, y_test), f_to_id = get_data_sets(ad)
    A_positive = []
    for x in range(len(y_train)):
        if y_train[x]:
            A_positive.append(A_train[x])
    features = set(name + "::" + f for a in A_positive for name in F_SETNAMES if name in ad[a] for f in ad[a][name])
    print(len(features))


if __name__ == '__main__':
    ad = load_articledict()
    # feature_count_positive(ad)
    # crossvalidation_search_decision_tree(ad)
    df = train_decisiontree_exploration(ad)
    ax = df.plot.scatter(x="FPR", y="TPR", style="x", c="purple")
    df = train_decisiontree_exploration(ad, splitnr=2000)
    df.plot.scatter(x="FPR", y="TPR", ax=ax, style="x", c="orange")
    df = train_decisiontree_exploration(ad, splitnr=3000)
    df.plot.scatter(x="FPR", y="TPR", ax=ax, style="x", c="red")
    df = train_decisiontree_exploration(ad, undersam=True, splitnr=3000)
    df.plot.scatter(x="FPR", y="TPR", ax=ax, style="o", c="purple")
    df = train_decisiontree_exploration(ad, oversam=True, splitnr=3000)
    df.plot.scatter(x="FPR", y="TPR", ax=ax, style="o", c="orange")
    df = train_decisiontree_exploration(ad, oversam=True, undersam=True, splitnr=3000)
    df.plot.scatter(x="FPR", y="TPR", ax=ax, style="o", c="red")
    plt.show()
    df.to_csv(DATAP + "/dct_kbest.csv")
