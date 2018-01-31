from mine.dbpedia import articles_with_property,CLURI,CFFURI

def check_infobox(langdict):
    print("Checking Infobox properties")
    propertylist = [
        "<http://dbpedia.org/property/dialects>",
        "<http://dbpedia.org/property/paradigm>",
        "<http://dbpedia.org/property/typing>",
        "^<http://dbpedia.org/ontology/language>",
        "^<http://dbpedia.org/property/language>",
        "^<http://dbpedia.org/ontology/programmingLanguage>"
        ]
    for p in propertylist:
        cls = articles_with_property(CLURI, 0, 6, p)
        cffs = articles_with_property(CFFURI, 0, 6, p)
        for cl in langdict:
            langdict[cl][p] = int(cl in cls | cffs)
    return langdict