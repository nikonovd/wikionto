import json
from mine.wiki import getlinks
from data import DATAP
from multiprocessing import Pool


def load_lists():
    with open(DATAP + "/lists.json", "r") as f:
        return json.loads(f.read())

def mine_list_links():
    lists = load_lists()
    pool = Pool(processes=10)
    list_links = pool.map(get_list_links, lists)

    with open(DATAP + "/listlinks.json", "w+") as out:
        json.dump(list_links, out, indent=4)


def get_list_links(list):
    print("Retrieving for %s"%list)
    links = getlinks(list)
    return list, links


if __name__ == "__main__":
    mine_list_links()