from check.langdictcheck import LangdictCheck
from data import KEYWORDS


class SumKeyWords(LangdictCheck):
    def check(self,langdict):
        print("Checking summary for keywords")
        for cl in langdict:
            summary = langdict[cl]["Summary"]
            if any(word in summary.lower() for word in KEYWORDS):
                langdict[cl]["PlainTextKeyword"] = 1
            else:
                langdict[cl]["PlainTextKeyword"] = 0
        return langdict


if __name__ == '__main__':
    SumKeyWords().solo()
