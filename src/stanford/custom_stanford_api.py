import requests
import json
from requests.sessions import Session
from http.client import HTTPConnection
import logging


class StanfordCoreNLP:
    """
    Modified from https://github.com/smilli/py-corenlp
    """

    def __init__(self, url, session=Session()):
        self.server_url = url
        self.session = session
        self.session.trust_env = False

    def annotate(self, text, annotators="tokenize,ssplit,pos", pattern=None, runnr=1):
        assert isinstance(text, str)

        properties = {
            "annotators": annotators,
            # Setting enforceRequirements to skip some annotators and make the process faster
            "enforceRequirements": "true",
            'timeout': 6000000000000,
            'tokenize.options': 'untokenizable=noneDelete'
        }
        params = dict()
        params['properties'] = str(properties)

        if pattern is not None:
            params['pattern'] = pattern

        try:
            with self.session.get(self.server_url) as req:
                data = text.encode('utf8')
                rs = Session()
                rs.trust_env = False
                r = rs.post(
                    self.server_url, params=params, data=data, headers={'Connection': 'close'})

                if r.status_code == 500:
                    print(r.content)
                    raise Exception500
                output = r.json()
                # output = json.loads(r.text, encoding='utf-8', strict=True)
        except:
            print("Caught Exception")
            if runnr > 10:
                print(text.encode('utf8'))
                return {}
            else:
                return self.annotate(text, properties, runnr=runnr+1)
        return output

def index_keyvalue_to_key(parsedict_list):
    result = dict()
    for parsedict in parsedict_list:
        d = parsedict
        index = parsedict["index"]
        del d["index"]
        result[index] = d
    return result

class Exception500(Exception):
    pass
