from mine.dbpedia import get_properties,CLURI,CFFURI
from data import DATAP,CLDEPTH, CFFDEPTH
from json import load, dump


def add_properties(langdict):
    print("Mining article properties")
    langdict = get_properties(CLURI, 0, CLDEPTH, langdict)
    langdict = get_properties(CFFURI, 0, CFFDEPTH, langdict)
    return langdict


if __name__ == '__main__':
    with open(DATAP + '/articledict.json', 'r', encoding='utf8') as f:
        langdict = load(f)
        langdict = add_properties(langdict)
    with open(DATAP + '/articledict.json', 'w', encoding='utf8') as f:
        dump(langdict, f, indent=2)
        f.flush()
        f.close()