# from mine.wikidata import get_computer_languages, get_computer_formats
# from mine.yago import get_artificial_languages
from json import load, dump
import time

# UTIL
DATAP = "/Users/dnikonov/Uni/wikionto/data/mined"
AP = DATAP + "/articledict.json"


def load_articledict():
    return load(open(AP, "r", encoding="utf-8"))


def valid_article(a, ad):
    return not ad[a]["IsStub"] and not ad[a]["DeletedFromWikipedia"] and not ad[a]["NotStandalone"] \
           and "List_of" not in a and "Comparison_of" not in a


def save_articledict(ad):
    with open(AP, "w", encoding="utf-8") as f:
        dump(ad, f)


def backup_articledict(ad):
    with open(DATAP+"/articledict_backup.json", "w", encoding="utf-8") as f:
        dump(ad, f)


def load_catdict():
    return load(open(DATAP + "/catdict.json", "r", encoding="utf-8"))


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def start_time():
    return time.time()


def stop_time(start_time):
    print(hms_string(time.time() - start_time))


def flat_list(l):
    return [item for sublist in l for item in sublist]


# SCOPING
DEPTH = 8
ROOTS = ["Category:Computing_platforms", "Category:Software"]

FEATURE_SETNAMES = ["DbpediaInfoboxTemplate", "URL_Braces_Words", "COPHypernym", "Lemmas", "Wikipedia_Lists"]  # Lemmas,
INDICATORS = ["PositiveInfobox", "URLBracesPattern", "In_Wikipedia_List", "PlainTextKeyword", "POS", "COP",
              "wikidata_CL"]

# CONFIG FOR INDICATORS
# - for POS and URLBracesPattern
KEYWORDS = ['software', 'system', 'application', 'framework', 'api', 'programming']

# - stretched keywords resulting which maybe hint at languages. Here, maybe means that we subjectively know that such
#   software has its own language, but! we cannot objectively present proof in the summary.
XKEYWORDS = [['file', 'type'], ['template', 'engine'], ['templating', 'system'], ['build', 'tool'],
             ['template', 'system'], ['theorem', 'prover'], ['parser', 'generator'], ['typesetting', 'system']]

# - infobox indication
POSITIVETEMPLATES = ["infobox_software", "infobox_os"]

# - Wikipedia Lists
# LIST_ARTICLES = retrievelists()

# CONFIG FOR NEGATIVE INDICATION (EXPLORATION USE)
# - Negative categories
NOISY_CATS = ['Category:Statistical_data_types', 'Category:Knowledge_representation',
              'Category:Propositional_attitudes‎',
              'Category:Theorems']
# - Note that all other infoboxes are negative. There are about 600 different templates used in the scope.
NEUTRALTEMPLATES = ["infobox", "infobox_windows_component", "infobox_software_license", "infobox_programming_language", "infobox_file_format", "infobox_os_component", "infobox_web_browser", "infobox_website", "infobox_file_system", "infobox_filesystem"]

# - Negative URL keywords
EX_URL_KEYWORD = ["List_of", "comparison", "Comparison"]
EX_URLBRACE_KEYWORD = ["song", "video_game", "TV_series"]
