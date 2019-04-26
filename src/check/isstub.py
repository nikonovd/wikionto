from check.abstract_check import ArtdictCheck
from mine.dbpedia import get_all_templates
from mine.dbpedia import to_uri
from mine.wiki import wiki_request
from multiprocessing import Pool
from data import flat_list


def chunk(l, n):
    chunks = []
    for i in range(0, len(l), n):
        chunks.append(l[i:i + n])
    return chunks


class IdentifyStubs(ArtdictCheck):

    def check(self, articledict):
        print("Checking for stubs...")
        qresult = get_all_templates(root=to_uri("Category:Computing_platforms"))
        qresult.update(get_all_templates(root=to_uri("Category:Software")))
        for a in articledict:
            articledict[a]["IsStub"] = 0
        for title, templates in qresult.items():
            if title in articledict and any("-stub" in t.lower() for t in templates):
                articledict[title]["IsStub"] = 1
        return articledict


class IdentifyStubsWikiApi(ArtdictCheck):

    def check(self, articledict):
        print("Checking for stubs using wikimedia api...")
        for a in articledict:
            articledict[a]["IsStub"] = 0

        titles = list(articledict.keys())
        print(len(titles))
        titlechunks = chunk(titles, 15)
        pool = Pool(processes=10)
        stub_checks = pool.map(self.check_stub, titlechunks)

        stub_checks = dict(list(set(flat_list(stub_checks))))

        print("All processed, writing dict.")

        for article in stub_checks:
            articledict[article]["IsStub"] = stub_checks[article]

        return articledict

    def check_stub(self, titlechunk):
        print("Checking %s" % titlechunk)
        titlesstring = '|'.join(titlechunk)
        # print(titlesstring)
        params = {
            "titles": titlesstring,
            "prop": "templates",
            "tllimit": 500,
            "tlnamespace": 10
        }
        response = wiki_request(params)
        if response is None:
            print("No response")
            return []
        if "continue" in response:
            print("continue...")
        normalized = {}
        if "normalized" in response["query"]:
            normalized = {normentry["to"]: normentry["from"] for normentry in response["query"]["normalized"]}

        result = []
        for pageentry in response["query"]["pages"]:
            title = pageentry["title"]
            if title in normalized:
                title = normalized[title]
            if "templates" not in pageentry:
                continue
            for templateentry in pageentry["templates"]:
                if "-stub" in templateentry["title"].lower():
                    result.append((title, 1))
                else:
                    result.append((title, 0))

        print("...done.")
        return list(set(result))



if __name__ == '__main__':
    IdentifyStubsWikiApi().solo()
