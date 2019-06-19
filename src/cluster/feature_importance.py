import pandas as pd
from data import DATAP
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.utility.construct_W import construct_W


def calc_feature_importance(data):
    print("Calculating feature importance matrix...")
    X = laplacian_score(data.values)
    # X = SPEC.feature_ranking(X, style=0)
    d_spec = pd.DataFrame([X], columns=data.columns.values, index=["SPEC"])
    d_spec = d_spec.sort_values("SPEC", axis=1, ascending=True)
    d_spec.to_csv(DATAP + "/cluster/seed_laplacian.csv")

    return d_spec


def load_feature_importance_matrix():
    print("Loading feature importance matrix...")
    matrix = pd.read_csv(DATAP + "/cluster/seed_SPEC.csv")
    # dropped_columns = [c for c in matrix.columns.values[1:] if c.split("::")[0] not in FEATURE_NAMES]
    # matrix = matrix.drop(labels=dropped_columns, axis=1)

    return matrix


def slice_n_relevant_features(data, feature_importance, n):
    print("Selecting the %s most relevant features..." % str(n))
    print(data.shape)
    print(feature_importance.shape)
    relevant_columns = feature_importance.iloc[0, 0:n+1].index.values[1:]
    return data.loc[:, relevant_columns]


def laplacian_score(data):
    W = construct_W(data)
    return lap_score(data, W=W)