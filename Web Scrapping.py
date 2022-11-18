#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


reviewlist = []


# In[3]:


def get_soup(url):
    link = requests.get(url)
    soup = BeautifulSoup(link.text,'html.parser')
    return soup


# In[4]:


def get_reviews(soup):
    reviews = soup.find_all('div',{'data-hook':'review'})
    try:
        for item in reviews:
            review = {
                'Text' : item.find('span',{'data-hook':'review-body'}).text.strip(),
                }
            reviewlist.append(review)
    except:
        pass


# In[5]:


for x in range(1,600):
    soup = get_soup(f'https://www.amazon.in/New-Apple-iPhone-11-64GB/product-reviews/B08L8C1NJ3/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li',{'class':'a-disabled a-last'}):
        pass
    else:
        break


# In[6]:


df = pd.DataFrame(reviewlist)
df.to_csv('reviews.csv', index=False)


# In[ ]:




