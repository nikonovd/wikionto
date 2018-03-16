from json import load

def check_semantic_distance(langdict):
    print("Checking semantic distance")
    f = open('data/catdict.json', 'r',encoding="UTF8")
    catdict = load(f)
    for cl in langdict: 
        cats = langdict[cl]["cats"]
        total = len(cats)
        reachable_cats = 0
        for cat in cats:
            if cat in catdict:
                reachable_cats += 1
            else:
                print(cat)
        langdict[cl]["SemanticDistance"] = total - (reachable_cats*2)
        langdict[cl]["SemanticallyRelevant"] = int(langdict[cl]["SemanticDistance"] < 1)
        langdict[cl]["NumberOfCategories"] = total
    return langdict

if __name__ == '__main__':
    import json
    with open('../data/langdict.json', 'r',encoding="UTF8") as f: 
        langdict = json.load(f)
        langdict = check_semantic_distance(langdict)
        f.close()
    with open('../data/langdict.json', 'w',encoding="UTF8") as f:
        json.dump(obj=langdict, fp=f, indent=2)
        f.flush()
        f.close()