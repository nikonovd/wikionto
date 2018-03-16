from check.gitseed import check_gitseed
from check.infobox import check_infobox
from check.hypernym_dbpedia import check_purlHypernymLanguage
from check.hypernym_stanford import check_stanford
from check.hypernym_wordnet import check_wordnet_hypernym
from check.semantic_distance import check_semantic_distance
from check.article_name_pattern import check_article_name
from check.empty_category import check_empty_cat
from json import dump, load
from check.cat_name_pattern import check_cat_name
from check.eponymous_cat import check_eponymous

from mine.miner import mine

mine()

with open('data/langdict.json', 'r',encoding="UTF8") as f: 
    langdict = load(f)
    langdict = check_gitseed(langdict)
    langdict = check_infobox(langdict)
    langdict = check_purlHypernymLanguage(langdict)
    langdict = check_stanford(langdict)
    langdict = check_wordnet_hypernym(langdict)
    langdict = check_semantic_distance(langdict)
    langdict = check_article_name(langdict)
    f.close()

with open('data/langdict.json', 'w',encoding="UTF8") as f:
    dump(obj=langdict, fp=f, indent=2)
    f.flush()
    f.close()
    
with open('data/catdict.json', 'r',encoding="UTF8") as f: 
    catdict = load(f)
    catdict = check_empty_cat(catdict,langdict)
    catdict = check_cat_name(catdict)
    catdict = check_eponymous(catdict,langdict)
    f.close()

with open('data/catdict.json', 'w',encoding="UTF8") as f:
    dump(obj=catdict, fp=f, indent=2)
    f.flush()
    f.close()