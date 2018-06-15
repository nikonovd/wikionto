from check.langdictcheck import LangdictCheck
from data import DATAP
from json import load


class Tiobe(LangdictCheck):

    def check(self, langdict):
        print("Checking TIOBE index")
        f = open(DATAP+"/TIOBE_index_annotated.json", 'r', encoding="UTF8")
        tiobedict = load(f)
        f.close()
        for cl in langdict:
            langdict[cl]["TIOBE"] = 0
        for tl, tld in tiobedict.items():
            if "recalledAs" in tld:
                rec = tld["recalledAs"]
                if rec in langdict:
                    langdict[rec]["TIOBE"] = 1
                else:
                    print(rec)
            elif tld["recall"]==1:
                langdict[tl]["TIOBE"] = 1
        return langdict


if __name__ == '__main__':
    Tiobe().solo()