from data import load_articledict


def analyze_feature_frequency(ad, **kwargs):
    freq = dict()

    if "F_SetNames" not in kwargs:
        kwargs["F_SetNames"] = []
    F_SetNames = kwargs["F_SetNames"]

    if "relevant_features" not in kwargs:
        kwargs["relevant_features"] = []
    relevant_features = kwargs["relevant_features"]

    for a in ad:
        for F_Name in F_SetNames:
            if F_Name not in ad[a]:
                continue
            for f in ad[a][F_Name]:
                if F_Name + "::" + f not in freq:
                    freq[F_Name + "::" + f] = 0
                freq[F_Name + "::" + f] += 1

        for feature in relevant_features:
            F_SetName, F_Name = feature.split("::")
            if F_SetName not in ad[a]:
                continue
            if F_Name in ad[a][F_SetName]:
                if feature not in freq:
                    freq[feature] = 0
                freq[feature] += 1
    return freq
