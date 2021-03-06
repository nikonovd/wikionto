from data import DATAP, ROOTS
from pandas import DataFrame, Series
from json import load

ton = 9
f = open(DATAP + '/catdict.json', 'r', encoding="UTF8")
catdict = load(f)
df = DataFrame(columns=['Cat', 'Seed Cat', 'SL Cat', "NoSL Cat", 'SL %'],
               index=['Formal languages', 'Computer file formats'])
for c in ROOTS:
    cats = [cat for cat in catdict if c + "Depth" in catdict[cat]]
    cats_withseed = [cat for cat in cats if catdict[cat]["#Seed"] > 0]
    cats_withsl = [cat for cat in cats if catdict[cat]["#Positive"] > 0]
    cats_nosl = [cat for cat in cats if catdict[cat]["#Positive"] == 0]
    percentage = 100 * (len(cats_withsl) / len(cats))

    df.loc[c.replace("Category:", "").replace("_", " ")] = Series({'Cat': len(cats), 'Seed Cat': len(cats_withseed),
                                                                   'SL Cat': len(cats_withsl),
                                                                   'NoSL Cat': len(cats_nosl),
                                                                   'SL %': percentage})
    print("----")
print(df.to_latex())
