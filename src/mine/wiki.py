import requests
from json.decoder import JSONDecodeError
from requests.sessions import Session
from json import dumps

URL = "http://en.wikipedia.org/w/api.php"
HEADER = {'User-Agent': 'WikiOnto'}


def wiki_request(params, action="query"):
    params['format'] = 'json'
    params['formatversion'] = 2
    params['action'] = action
    params['utf8'] = ''
    try:
        s = Session()
        s.trust_env = False
        r = s.get(URL, params=params, headers=HEADER).json()
    except requests.ConnectionError as cer:
        print("Connection Error")
        print(cer)
        r = wiki_request(params)
    except JSONDecodeError:
        return None
    return r


def getfirstsentence(name):
    params = {'prop': 'extracts'
        , 'exintro': ''
        , 'exsentences': '1'
        , 'titles': name
        , 'redirects': '1'}
    wikijson = wiki_request(params)
    sentence = next(iter(wikijson["query"]["pages"].values()))["extract"]
    return sentence


def getsubcats(title):
    params = {'cmlimit': '500'
        , 'list': 'categorymembers'
        , 'cmtype': 'subcat'
        , 'cmtitle': title
        , 'redirects': '1'}
    wikijson = wiki_request(params)
    members = wikijson["query"]["categorymembers"]
    subcats = [member["title"].replace(" ", "_") for member in members]

    return subcats


def getarticles(title):
    params = {'cmlimit': '500'
        , 'list': 'categorymembers'
        , 'cmtype': 'page'
        , 'cmtitle': title
        , 'redirects': '1'}
    wikijson = wiki_request(params)
    members = wikijson["query"]["categorymembers"]
    articles = [m["title"].replace(" ", "_") for m in members]
    return articles


def getcategories(title):
    params = {'prop': 'categories'
        , 'titles': title
        , 'redirects': '1'}
    wikijson = wiki_request(params)
    members = next(iter(wikijson["query"]["pages"].values()))["categories"]
    categories = [m["title"].replace(" ", "_") for m in members]
    return categories


def getcontent(revid):
    params = {'prop': 'revisions'
        , 'rvprop': 'content'
        , 'revids': revid}
    wikijson = wiki_request(params)
    if wikijson is None:
        return None
    try:
        return wikijson
    except KeyError:
        return None


def getlinks(title):
    params = {
        'prop': 'links',
        'titles': title
    }

    wikijson = wiki_request(params)
    # print(wikijson)
    links = []
    try:
        # if 'missing' in next(iter(wikijson["query"]["pages"])):
        #     return links
        while True:
            if "query" not in wikijson:
                print("None at query " + title)
                return links
            if "pages" not in wikijson["query"]:
                print("None at pages " + title)
                return links
            if wikijson["query"]["pages"] is None:
                print("None at values " + title)
                return links
            nextpages = next(iter(wikijson["query"]["pages"]))
            if 'links' not in nextpages:
                print("None at links " + title)
                return links
            ls = nextpages['links']
            for l in ls:
                links.append(l['title'])
            if 'continue' not in wikijson:
                break
            plcontinue = wikijson['continue']['plcontinue']
            params['plcontinue'] = plcontinue
            wikijson = wiki_request(params)
        return links
    except KeyError:
        return links


def get_redirect(title):
    params = {
        'titles': title,
        'redirects': '1'
    }
    try:
        wikijson = wiki_request(params)

        if "query" not in wikijson:
            print("None at query " + title)
            return []
        if "pages" not in wikijson["query"]:
            print("None at pages " + title)
            return []
        if wikijson["query"]["pages"] is None:
            print("None at values " + title)
            return []

        target_page = next(iter(wikijson["query"]["pages"]))

        if "title" not in target_page:
            print("None at title")
            return []

        print("%s -> %s" % (title, target_page["title"]))

        return target_page["title"]
    except KeyError:
        print("Redirect resolved in KeyError for %s" % title)
        return []


def get_summary_links(title):
    params = {
        'page'   : title,
        'section': 0
    }

    try:
        wikijson = wiki_request(params, action="parse")

        # print(wikijson)
        raw_links = wikijson["parse"]["links"]

        return [l["title"] for l in raw_links]
    except KeyError:
        print("KeyError for %s." % title)
        return []


def getlinks_multi(titles):
    params = {'prop': 'links',
              'titles': titles}
    wikijson = wiki_request(params)
    links = []
    return links


def get_infobox(pair):
    l = pair[0]
    rev = pair[1]
    text = getcontent(rev).lower()
    if '{{infobox' not in text:
        return l, rev, []
    parts = text.split('{{infobox')
    ibs = []
    for x in range(1, len(parts)):
        p = parts[x]
        name = p.split('|')[0].replace('\\n', '').strip()
        ibs.append(name)
    return l, rev, ibs


if __name__ == "__main__":
    from data import load_articledict

    ad = load_articledict()
    titles = list(ad.keys())
    titles = titles[:100]
    c = getlinks_multi(titles)
