from nltk.tag.stanford import CoreNLPPOSTagger
from nltk.parse.corenlp import CoreNLPDependencyParser
from mine.dbpedia import articles_with_summaries,CFFURI,CLURI

def check_stanford(langdict):
    print("Checking Hypernym with Stanford")
    clarticles = articles_with_summaries(CLURI,0,6)
    cffarticles = articles_with_summaries(CFFURI, 0, 6)
    clarticles.update(cffarticles)
    zipped_art_sum = list(zip(*clarticles.items()))
    split_art_sum = map(lambda s: s.split(),zipped_art_sum[1])
    pos_tagged = CoreNLPPOSTagger(url='http://localhost:9000').tag_sents(list(split_art_sum))
    cls_pos_tagged = dict(zip(zipped_art_sum[0],pos_tagged))
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    dep_parsed = list(map(lambda s: dep_parser.raw_parse(s),zipped_art_sum[1]))
    cls_dep_parsed = dict(zip(zipped_art_sum[0],dep_parsed))
    
    for cl in langdict:
        if cl not in cls_pos_tagged:
            langdict[cl]["StanfordPOSHypernym"] = 0
            langdict[cl]["StanfordCOPHypernym"] = 0
            langdict[cl]["Summary"]="No Summary!"
        else: 
            langdict[cl]["Summary"] = clarticles[cl]
            langdict[cl]["StanfordPOSHypernym"] = pos_language(cls_pos_tagged[cl])
            langdict[cl]["StanfordCOPHypernym"] = cop_language(cls_dep_parsed[cl])
    return langdict

def pos_language(tagged):
    vbzs = {(w,p) for (w,p) in tagged if (p=="VBZ") & ("is" in w)}
    nns = {(w,p) for (w,p) in tagged if (p=="NN") & (("language" in w.lower()) | ("format" in w.lower()))} 
    return int(bool(vbzs) & bool(nns))

def cop_language(tagged):
    parse, = tagged
    for subj, dep, obj in parse.triples():
        if (subj[1]=='NN')&(subj[0] in ['language','format','dsl','dialect']) & (dep=='cop') & (obj==('is','VBZ')):
            return 1
    return 0

def run_solo():
    import json
    with open('../data/langdict.json', 'r',encoding="UTF8") as f: 
        langdict = json.load(f)
        langdict = check_stanford(langdict)
        f.close()
    with open('../data/langdict.json', 'w',encoding="UTF8") as f:
        json.dump(obj=langdict, fp=f, indent=2)
        f.flush()
        f.close()

#run_solo()
