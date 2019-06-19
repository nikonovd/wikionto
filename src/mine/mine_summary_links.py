from mine.wiki import get_summary_links, get_redirect
from data import load_seedlist, DATAP, load, load_articledict, save_articledict
from multiprocessing import Pool
from mine.util import flatten
import json


def mine_summary_links():
    seed = load_seedlist()
    titles = list(seed.keys())

    pool = Pool(processes=10)
    summary_links = dict(pool.map(article_summary_links, titles))

    with open(DATAP + "/seed_summary_links.json", "w+") as out:
        json.dump(summary_links, out, indent=4)


def article_summary_links(title):
    target_title = get_redirect(title)
    print("Retrieving for %s..." % target_title)
    links = get_summary_links(target_title)

    links = sanitize_links(links)
    links = resolve_redirects(links)

    return title, links


def bad_link(link):
    return any([
        "Wikipedia:" in link,
        "Template:" in link,
        "Template talk:" in link,
        "Help:" in link,
        "Category:" in link,
        "Talk:" in link,
        "Software categories" in link,
        "Software developer" in link,
        "Software license" in link,
        "Software release life cycle" in link,
    ])


def sanitize_links(seed_links):
    return [l for l in seed_links if not bad_link(l)]


def resolve_redirects(seed_links):
    resolved_tuples = [resolve_redirect(l) for l in seed_links]
    return [t[1] for t in resolved_tuples]


def resolve_redirect(title):
    target = get_redirect(title)
    return title, target


if __name__ == "__main__":
    # mine_summary_links()
    # seed_links = json.load(open(DATAP + "/seed_summary_links.json", "r"))
    #
    # seed_links = {
    #     s[0]: [l for l in s[1] if not bad_link(l)] for s in seed_links.items()
    # }
    #
    # json.dump(seed_links, open(DATAP + "/seed_summary_links.json", "w+"), indent=4)

    ad = load_articledict()
    seed_links = json.load(open(DATAP + "/seed_summary_links.json", "r"))

    for s in seed_links:
        ad[s]["InternalWikiLinks"] = seed_links[s]

    save_articledict(ad)
