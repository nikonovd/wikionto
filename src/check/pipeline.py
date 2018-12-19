from check.seed import Seed
from check.hypernym_dbpedia import DbpediaHyp
from check.pos_firstsentence import HypNLPSent
from check.cop_firstsentence import COPFirstSentence
from check.summary_keywords import SumKeyWords
from check.url_pattern import URLPattern
from check.infobox_dbpedia_existence import InfoboxDbEx
from check.lists_of import WikiList
from check.semantic_distance import SemDist, SemDistCat
from check.childtest import ChildCheck
from check.url_pattern_cat import CategoryURLPattern
from check.eponymous_cat import EponymousCat
from check.cycles import Cycle
from check.foldoc import FoldocTopic


def article_indicators():
    indicators = [Seed, InfoboxDbEx, URLPattern, WikiList, FoldocTopic, SumKeyWords, DbpediaHyp, HypNLPSent, COPFirstSentence, SemDist]
    for i in indicators:
        i().solo()


def cat_indicators():
    catindicator = [CategoryURLPattern, EponymousCat, ChildCheck, SemDistCat, Cycle]
    for c in catindicator:
        c().solo()


if __name__ == '__main__':
    #mine()
    article_indicators()
    cat_indicators()
