from multiprocessing import Pool
from nltk.parse.corenlp import CoreNLPDependencyParser
from requests.exceptions import HTTPError
from mine.dbpedia import articles_with_summaries
from data import DATAP
from json import dump, load
from json.decoder import JSONDecodeError
import operator
import pandas
import matplotlib.pyplot as plt

def plot_top():
    f = open(DATAP + '/langdict.json','r',encoding="UTF8")
    headers = ['word', '#articles']
    df = pandas.read_csv(f, names=headers)
    df = df[df['#articles']>0].sort_values(by='#articles')
    print(df)
    df.tail(10).plot(x="word", y="#articles",kind='bar', color='royalblue')
    # beautify the x-labels
    # plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    plot_top()