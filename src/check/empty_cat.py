from json import load, dump
from data import DATAP


def check_empty_cat(catdict, langdict):
    print("Checking for categories with no relevant articles")
    for cat in catdict:
        if "articles" not in catdict[cat]:
            catdict[cat]["ChildTest"] = 0
        else:
            count = 0
            for cl in catdict[cat]["articles"]:
                if lang_succeeds_at_none(cl, langdict):
                    count += 1
            catdict[cat]["NoLangRatio"] = count
            catdict[cat]["#articles"] = len(catdict[cat]["articles"])
            if count > len(catdict[cat]["articles"]) / 2:
                catdict[cat]["ChildTest"] = 0
            else:
                catdict[cat]["ChildTest"] = 1
    return catdict


def lang_succeeds_at_none(lang, langdict):
    fs = ["GitSeed", "TIOBE", "DbpediaHypernym", "PlainTextKeyword", "POS", "COP", "URLPattern",
          "URLBracesPattern", "Infobox programming language", "Infobox file format",
          "Infobox software", "wikidata_CL", "yago_CL"]
    flag = bool(sum(map(lambda p: p in langdict[lang] and langdict[lang][p], fs)))
    sr = langdict[lang]['Semantic_Distance'] > 1
    sim = langdict[lang]['Semantic_Distance'] < 0.1
    return flag | sr | sim


if __name__ == "__main__":
    f = open(DATAP + '/langdict.json', 'r', encoding="UTF8")
    langdict = load(f)
    with open(DATAP + '/catdict.json', 'r', encoding="UTF8") as f:
        catdict = load(f)
        catdict = check_empty_cat(catdict, langdict)
        f.close()
    with open(DATAP + '/catdict.json', 'w', encoding="UTF8") as f:
        dump(obj=catdict, fp=f, indent=2)
        f.flush()
        f.close()
