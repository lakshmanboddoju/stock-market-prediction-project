#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from newsapi import NewsApiClient

key = '10d685a5802043e09cb18d63e01e6a30'
params = {}
api = NewsApiClient(api_key=key)
sources = api.get_sources()
articles = api.get_top_headlines(sources="the-new-york-times")

import sys, csv, json
reload(sys)
sys.setdefaultencoding('utf8')


import requests

class APIKeyException(Exception):
    def __init__(self, message): self.message = message 

class InvalidQueryException(Exception):
    def __init__(self, message): self.message = message 

class ArchiveAPI(object):
    def __init__(self, key=None):
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}' 
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None,):
        if not key: key = self.key
        if (year < 1882) or not (0 < month < 13):
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)
        url = self.root.format(year, month, key)
        r = requests.get(url)
        return r.json()


api = ArchiveAPI('d874704c3a91403a830c4542a5c95245')


years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for year in years:
    for month in months:
        mydict = api.query(year, month)
        file_str = 'data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str, 'w') as fout:
            json.dump(mydict, fout)
        fout.close()