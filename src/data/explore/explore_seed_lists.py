from data import DATAP, KEYWORDS
from json import load

f = open(DATAP + '/articledict.json', 'r', encoding="UTF8")
articledict = load(f)
for title in [a[0] for a in articledict if a[1]["Seed"] == 1]:
    if ("list" in title.lower() or "comparison" in title.lower()) and any(k in title for k in KEYWORDS):
        print(title)
