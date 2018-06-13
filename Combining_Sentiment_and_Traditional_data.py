#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import csv, json
import pandas as pd

with open('data/aapl_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    data_list = list(spamreader)

header = data_list[0] 
data_list = data_list[1:] 

data_list = np.asarray(data_list)

selected_data = data_list[:, [0, 4, 6]]

df = pd.DataFrame(data=selected_data[0:,1:],
             index=selected_data[0:,0],
                                columns=['close', 'adj close'],
                                        dtype='float64')


df1 = df
idx = pd.date_range('12-29-2006', '12-31-2016')
df1.index = pd.DatetimeIndex(df1.index)
df1 = df1.reindex(idx, fill_value=np.NaN)

interpolated_df = df1.interpolate()
interpolated_df.count() 

interpolated_df = interpolated_df[3:]

date_format = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+%f"]
def try_parsing_date(text):
    for fmt in date_format:
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    raise ValueError('no valid date format found')




years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dict_keys = ['pub_date', 'headline'] 
articles_dict = dict.fromkeys(dict_keys)

type_of_material_list = ['blog', 'brief', 'news', 'editorial', 'op-ed', 'list','analysis']

section_name_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health']
news_desk_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health', 'foreign']

current_date = '2016-01-01'
from datetime import datetime


current_article_str = ''      

interpolated_df["articles"] = ''
count_articles_filtered = 0
count_total_articles = 0
count_main_not_exist = 0               
count_unicode_error = 0     
count_attribute_error = 0   
for year in years:
    for month in months:
        file_str = 'data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str) as data_file:    
            NYTimes_data = json.load(data_file)
        count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
        for i in range(len(NYTimes_data["response"]["docs"][:])):
            try:
                if any(substring in NYTimes_data["response"]["docs"][:][i]['type_of_material'].lower() for substring in type_of_material_list):
                    if any(substring in NYTimes_data["response"]["docs"][:][i]['section_name'].lower() for substring in section_name_list):
                        count_articles_filtered += 1
                        articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                        articles_dict['headline'] = articles_dict['headline']['main']
                        date = try_parsing_date(articles_dict['pub_date'])
                        if date == current_date:
                            current_article_str = current_article_str + '. ' + articles_dict['headline']
                        else:  
                            interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[current_date, 'articles'] + '. ' + current_article_str)
                            current_date = date
                            current_article_str = articles_dict['headline']
                        if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1): 
                            interpolated_df.set_value(date, 'articles', current_article_str)   
                        
            except AttributeError:
                count_attribute_error += 1
                try:
                    if any(substring in NYTimes_data["response"]["docs"][:][i]['news_desk'].lower() for substring in news_desk_list):
                            count_articles_filtered += 1
                            articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                            articles_dict['headline'] = articles_dict['headline']['main'] 
                            date = try_parsing_date(articles_dict['pub_date'])
                            if date == current_date:
                                current_article_str = current_article_str + '. ' + articles_dict['headline']
                            else:  
                                interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[current_date, 'articles'] + '. ' + current_article_str)
                                current_date = date
                                current_article_str = articles_dict['headline']
                            if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1): 
                                interpolated_df.set_value(date, 'articles', current_article_str)   
                
                except AttributeError:
                    pass
                pass
            except KeyError:
                print 'key error'
                count_main_not_exist += 1
                pass   
            except TypeError:
                print "type error"
                count_main_not_exist += 1
                pass
         

              
print count_articles_filtered 
print count_total_articles                     
print count_main_not_exist
print count_unicode_error

for date, row in interpolated_df.T.iteritems():   
    if len(interpolated_df.loc[date, 'articles']) <= 400:
        month = date.month
        year = date.year
        file_str = 'data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str) as data_file:    
            NYTimes_data = json.load(data_file)
        count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
        interpolated_df.set_value(date.strftime('%Y-%m-%d'), 'articles', '')
        for i in range(len(NYTimes_data["response"]["docs"][:])):
            try:
                
                articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                articles_dict['headline'] = articles_dict['headline']['main']      
                pub_date = try_parsing_date(articles_dict['pub_date'])
                if date.strftime('%Y-%m-%d') == pub_date: 
                    interpolated_df.set_value(pub_date, 'articles', interpolated_df.loc[pub_date, 'articles'] + '. ' + articles_dict['headline'])  
                
            except KeyError:
                print 'key error'
                pass   
            except TypeError:
                print "type error"
                pass


interpolated_df.to_pickle('data/pickled_ten_year_filtered_lead_para.pkl')  

interpolated_df.to_csv('data/sample_interpolated_df_10_years_filtered_lead_para.csv',
                       sep='\t', encoding='utf-8')

dataframe_read = pd.read_pickle('data/pickled_ten_year_filtered_lead_para.pkl')
