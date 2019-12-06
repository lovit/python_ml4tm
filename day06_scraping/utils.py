import re
import requests
import sys
import bs4
from bs4 import BeautifulSoup


print('Python version = {}'.format(sys.version_info))
print('BeautifulSoup version = {}'.format(bs4.__version__))

def get_soup(url, headers=None):
    try:
        r = requests.get(url, headers=headers).text
        return BeautifulSoup(r, 'lxml')
    except Exception as e:
        print(e)
        return None

normalize_pattern = re.compile('[\n\t]')
doublespace_pattern = re.compile('\s+')

def normalize(text):
    text = normalize_pattern.sub(' ', text)
    text = doublespace_pattern.sub(' ', text)
    return text.strip()
