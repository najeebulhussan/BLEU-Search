# import necessary libraries

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)
#Import all the dependencies
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import wikipedia

import plotly.plotly as py
from plotly.graph_objs import *

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import urllib.request
import re
from inscriptis import get_text
from pprint import pprint
from textblob import Word
import sys
import random
import nltk 
from nltk.corpus import wordnet 
import plotly.figure_factory as ff
import win32com.client as wincl
import win32api
import time
from selenium import webdriver
from splinter import Browser
import glob
from flask import jsonify
import pymongo
import queue
from threading import Thread
from nltk.util import ngrams
import os

speak = wincl.Dispatch("SAPI.SpVoice")
speak.Speak("")
##Checking 
def predict(image_path):
    """Use Xception to label image"""
    model = Xception(
    include_top=True,
    weights='imagenet')
    image_size = (299, 299)
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    plt.imshow(img)
    return decode_predictions(predictions, top=3)[0]

def get_number(s):
    newstr = ''.join((ch if ch in '0123456789.,/:%' else ' ') for ch in s)
    return newstr

def get_sentencenumber(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html)
    title = soup.find('h1').text
    print(title)
    text = get_text(html)
   # result1 = re.sub(r"\s+"," ", text, flags = re.I)
   #sentence=re.compile("[A-Z].*?[\.!?] ", re.MULTILINE | re.DOTALL )
   # a = sentence.findall(result1)
    a =   re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
   # a = sentence.findall(result1)
    content = []
    number=[]
    content.append({"Content":url,"Numbers" : title})
    for x in a:
        result = re.findall(r"\d+", x)
        if len(result) >0:
            s= re.sub(r"\s+"," ",get_number(x), flags = re.I).split(" ")
            #print(s)
            for ch in s:
                if ch not in ["",".",":","/","%",","," ","://",".,",".."]:
                    number.append(get_number(ch))
            content.append({"Content":x,"Numbers": number})
            number =[]
    
     # Check thread's return value
    df1 = pd.DataFrame(content,columns =['Content', 'Numbers'])
    if len(df1) > 0:
        data = df1.T.to_dict().values()
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        analyst = db.analyst
        db.drop_collection(analyst)
        analyst.insert_many(data,ordered=False)
        return df1.to_records().tolist()
    else:
        return None




### Preprocessing 
##

def pathtostring(pathnames):
    path2string = []
    for x in pathnames:    
        head, tail = os.path.split(x)
        #path2string.append(head[10:])
        path2string.append(os.path.splitext(tail)[0])
    return " ".join(list(set(path2string)))
    ##
def removeduplicates(string):
    words = string.lower().split()
    return " ".join(sorted(set(words), key=words.index))
## #Try to search info of searchstr thru Wikipedia to get information
def wiki(searchstr):
    results = wikipedia.search(searchstr,results=3)
    print(results)
    found_results=[]
    title=""
    url=""
    content=""
    image=[]
    ranking=100
    if len(results) >0:
        for x in results:
            try:
                result = wikipedia.page(x)
                if result.title != "":
                    title= result.title
                if result.url != "":
                    url = result.url
                if wikipedia.summary(x,sentences=2) != "":
                    content =  wikipedia.summary(x,sentences=2)
                if len(result.images) >0:
                    image = result.images[1]
                found_results.append({"Title":title,"Link":url,'Description':content,"Image":image,'Keyword':title,'Ranking':ranking})
            except wikipedia.exceptions.DisambiguationError as e:
                pass
            except KeyError:
                pass
    return found_results

# get searchstring by using N-gram to get combined words from 2 - i (i less than total length of search string) words in search string
def get_ngrams(text):
    total =[]
    n=2
    while n <=len(word_tokenize(text)):
        n_grams = ngrams(word_tokenize(text), n)
        n = n+1
        total = total + [' '.join(grams) for grams in n_grams]
    return total
#Get Ngram with lemmatize to singular and convert Verb to original before compare to get right indexes for search results
def get_ngrammatches(text):
    a=list(set(word_tokenize(text)+ [Word(wordlemma).lemma for wordlemma in word_tokenize(text)]+ [Word(wordlemma).lemmatize("v") for wordlemma in word_tokenize(text)]))
    total =[]
    n=1
    while n <=len(word_tokenize(text)):
        n_grams = ngrams(a, n)
        n = n+1
        total = total + [' '.join(grams) for grams in n_grams]
    return total

# Find 1 similar meaning of each word in searching string plus search string, equal len(searchstring)x2+1 

def word_synonymone(lines):
    words = lines.split()
    random_syn =[]
    if len(words)>1:
        random_syn.append(lines)
    for word_str in words:
        word_obj = Word(word_str)
        synonyms = []
        if len(wordnet.synsets(word_obj)) > 0: # in case do not have any synonyms
            for syn in wordnet.synsets(word_obj): 
                for l in syn.lemmas():
                    if l.name() != word_obj:
                        synonyms.append(l.name())
            random_syn.append((random.choice(synonyms)).replace("_","-"))
            random_syn.append(word_obj)
        else:
            random_syn.append(word_obj)
    return random_syn
# Find all similar meaning of each word in searching string plus search string 

def word_synonymall(lines):
    words = lines.split()
    synonyms = [] 
    for word_str in words:
        word_obj = Word(word_str)
        if len(wordnet.synsets(word_obj)) > 0: # In case do not have synonyms
            for syn in wordnet.synsets(word_obj): 
                for l in syn.lemmas():
                    if l.name() != word_obj:
                        synonyms.append(l.name().replace("_","-"))
                synonyms.append(word_obj)
        else:
            synonyms.append(word_obj)
    return list(set([x.lower() for x in synonyms])) #Convert to lower and Remove duplicated words

#### Find all synonyms but remove original words for ranking/indexing the results

def synonymall(keyword):
    synonyms = [] 
    for word_str in keyword:
        word_obj = Word(word_str)
        if len(wordnet.synsets(word_obj)) > 0: # In case do not have synonyms
            for syn in wordnet.synsets(word_obj): 
                for l in syn.lemmas():
                    if l.name() != word_obj:
                        synonyms.append(l.name().replace("_","-"))
    return list(set([x.lower() for x in synonyms])) #Convert to lower and Remove duplicated words

# Find 2 similar meaning of each word in searching string plus search string, equal len(searchstring)x3+1 
def word_synonymtwo(lines):
    try:
        df =pd.read_csv("static//js//Category//stopword.csv")
        stopwords=df["stopwords"].tolist()
        texts = ' '.join([word.lower() for word in lines.split()
              if word.lower() not in stopwords])
        words = texts.split()
        random_syn =[]
        if len(words)>1:
            #random_syn.append(texts)
            random_syn =get_ngrams(texts)
        for word_str in words:
            word_obj = Word(word_str)
            synonyms = []
            if len(wordnet.synsets(word_obj)) > 0: # in case do not have any synonyms
                for syn in wordnet.synsets(word_obj):
                    for l in syn.lemmas():
                        if l.name() != word_obj:
                            synonyms.append(l.name())
                        else:
                            synonyms.append(word_obj)
                for x in random.choices(synonyms,k=2):
                    random_syn.append(x.replace("_"," "))
                random_syn.append(word_obj)
            else:
                random_syn.append(word_obj)
        return list(set([x.lower() for x in random_syn]))
    
    except Exception as e:
        print(e)

#Remove special symbols and clean space from search Description
def removespecial(desc):
    removespecials = ["\n","\t"," ","`","'"]
    words = desc.split()
    ss=""
    for x in removespecials:
        desc.replace(x,"")
    for word in desc.split():
        ss=ss+" " + word
    return ss.strip()

### Process

##############
#Search only images
def parseimg(url,searchstr):
    resultsalt = []
    desc=""
    title=""
    sites=[]
    image_list=""
    linklist=[]
    try:
        r = requests.get(url,timeout=3)
        if r.status_code  == 200:
            soup = BeautifulSoup(r.content,"lxml")
            for x in soup.findAll('div'):
                ## Inside a
                for k in x.find_all('a'):
                    #print(removespecial(k.text))
                    if k is not None:
                        sites = k.get("href")
                    if k not in linklist:
                        for i in k.find_all('img'):
                            if i.get('src') is not None:
                                image_list = i.get('src')
                                desc = i.get('alt') + " " +removespecial(k.text)
                                if (image_list !=''and desc !=''):
                                    if image_list is not None and image_list.startswith("http")==False:
                                        image_list = url+image_list
                                    if sites is not None and sites.startswith("http")==False:
                                        sites = url+sites
                                    resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                                image_list=""
                                desc=""

                            elif i.get('data-img-src') is not None:
                                image_list = i.get('data-img-src')
                                desc = i.get('alt')+ " " +removespecial(k.text)
                                if (image_list !=''and desc !=''):
                                    if image_list is not None and image_list.startswith("http")==False:
                                        image_list = url+image_list
                                    if sites is not None and sites.startswith("http")==False:
                                        sites = url+sites
                                    resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                                image_list=""
                                desc=""

                            elif i.get("data-src") is not None:
                                image_list = i.get('data-src')
                                desc = i.get('data-alt') + " " +removespecial(k.text)
                                if (image_list !=''and desc !=''):
                                    if image_list is not None and image_list.startswith("http")==False:
                                        image_list = url+image_list
                                    if sites is not None and sites.startswith("http")==False:
                                        sites = url+sites
                                    resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                                image_list=""
                                desc=""
                            linklist.append(k)
##Missing Images in DIV with no hyperlink
            #In case of using DIV with data-src,src,data-img-src
                if x.get('src') is not None:
                    image_list = x.get('src')
                    desc = x.get('alt')
                    if (image_list !=''and desc !=''):
                        if image_list is not None and image_list.startswith("http")==False:
                            image_list = url+image_list
                        if sites is not None and sites.startswith("http")==False:
                            sites = url+sites
                        resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                    image_list=""
                    desc=""
                elif x.get('data-img-src') is not None:
                    image_list = x.get('data-img-src')
                    desc = x.get('alt')
                    if (image_list !=''and desc !=''):
                        if image_list is not None and image_list.startswith("http")==False:
                            image_list = url+image_list
                        if sites is not None and sites.startswith("http")==False:
                            sites = url+sites
                        resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                    image_list=""
                    desc=""
                elif x.get("data-src") is not None:
                    image_list = x.get('data-src')
                    desc = x.get('data-alt')
                    if (image_list !=''and desc !=''):
                        if image_list is not None and image_list.startswith("http")==False:
                            image_list = url+image_list
                        if sites is not None and sites.startswith("http")==False:
                            sites = url+sites
                        resultsalt.append({"Link":sites, "Image": image_list,'Description': desc})
                    image_list=""
                    desc=""
                image_list=""
                desc=""
                sites=""
                       
    except Exception as e:
        pass
    result = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in resultsalt)] #Remove duplicate elements in list results
    delindex=[]#Create delete index to remove unmatched 
    #print(result)
    searchkeys= " ".join(key for key in searchstr)
    for index in range(len(result)):
        if result[index]["Description"] is None:
            delindex.append(index)
        else:
            if matchimage(searchkeys,result[index]["Description"]) ==0:
                delindex.append(index)
    for index in sorted(delindex, reverse=True):
        del result[index]      # Deleting unmatches results
    print(len(result))
    return  result
#Use multithreading to increase speed of the search engine.if python 2 can use multiprocessing
def mainsearchimages(dataframefile,searchstr):
    df = pd.concat([pd.read_csv(open(f, 'r',encoding='utf8')) for f in dataframefile])
    #convert to list and remove duplicates
    urls=list(set(df["Link"].tolist()))
    searchstround = removeduplicates(searchstr+" " + pathtostring(dataframefile))
    searchwords = word_synonymtwo(searchstround)+wikipedia.search(searchstround,results=3)
    print(searchwords)
    que = queue.Queue()
    threads_list = list()
    wn.ensure_loaded()
    for i in urls:
        t = Thread(target=lambda q, arg1,arg2: q.put(parseimg(arg1,arg2)), args=(que, i,searchwords))
        t.daemon = True
        t.start()
        threads_list.append(t)
    # Join all the threads
    for t in threads_list:
        t.join()

    # Check thread's return value
    found_results=[]
    while not que.empty():
            found_results.extend(que.get())
    df1 = pd.DataFrame(found_results,columns =['Link', 'Description','Image'])
    if len(df1) > 0:
        dffinal = rankindeximages(searchstr,df1,"findoutimages").reset_index(drop=True)
        for index, row in dffinal.iterrows():
            dffinal.loc[[index],"Ranking"] =  int(countmatches(searchstr,row.Description))
        dffinal.sort_values('Ranking', ascending=False, inplace=True)
        dffinal.to_csv("static//js//Category//findoutimages.csv",index=False)
        data = dffinal.T.to_dict().values()
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        searchresultsimages = db.searchresultsimages
        db.drop_collection(searchresultsimages)
        searchresultsimages.insert_many(data)
        return dffinal.to_records().tolist()
    else:
        return None
###############
def parselinkimg(url,searchstr):
    #checkk = wikipedia.search(searchstr,results=3)
    found_results = []
    site=""
    desc=""
    title=""
    sites=[]
    image_list=""
    try:
        r = requests.get(url,timeout=3)
        print("URL chung")
        print(url)
        if r.status_code  == 200:
            print(url)
            #print("main check")
            soup = BeautifulSoup(r.content,"lxml")
            #print("main check1")
            #print(word_synonymtwo(searchstr))
            for i in searchstr:
                print(i)
                links = lambda tag: getattr(tag, 'name', None) == 'a' and 'href' in tag.attrs and i.lower() in tag.get_text().lower()
                results = soup.find_all(links)
                
                #print(results)
                if len(results) > 0:
                    for index in range(len(results)):
                        #print(results[index])
                        #Remove duplicate link with same keyword, in case diff keywords then keep the link and add keywords
                        if sites.count(results[index].get('href'))==0:
                            #In case the links not begin with http,https, www
                            if (results[index].get('href')[0:4].lower() != "http" and results[index].get('href')[0:3].lower() != "www" and results[index].get('href')[0:4].lower() != "//ww" and results[index].get('href')[0:1]!="."):
                             
                                site = url+results[index].get('href')
                                desc = removespecial(results[index].text)
                                if results[index].get('title') is not None: # In case don have title tage in website
                                    title= removespecial(results[index].get('title'))
                                #Image will keep in Img tag with Src or DIV with data-src or both 
                                if results[index].find('div') is not None:
                                    photo = results[index].find(lambda tag: tag.name == 'div' and 'data-src' in tag.attrs)
                                    if photo is not None:
                                        image_list = photo.get("data-src")
                                if results[index].find('img') is not None:
                                    if results[index].find('img').get('src') is not None:
                                        image_list = results[index].find('img').get('src')
                                    elif results[index].find('img').get('data-img-src') is not None:
                                        image_list = results[index].find('img').get('data-img-src')
                                    elif results[index].find('img').get("data-src") is not None:
                                        image_list = results[index].find('img').get('data-src')
                            # iN CASE the link begin with . then remove . and add the home link of the website
                            elif (results[index].get('href')[0:1]=="."):
                                
                                site = url+results[index].get('href')[1:]
                                desc = removespecial(results[index].text)
                                if results[index].get('title') is not None:
                                    title= removespecial(results[index].get('title'))
                                if results[index].find('div') is not None:
                                    photo = results[index].find(lambda tag: tag.name == 'div' and 'data-src' in tag.attrs)
                                    if photo is not None:
                                        image_list = photo.get("data-src")
                                if results[index].find('img') is not None:
                                    if results[index].find('img').get('src') is not None:
                                        image_list = results[index].find('img').get('src')
                                    elif results[index].find('img').get('data-img-src') is not None:
                                        image_list = results[index].find('img').get('data-img-src')
                                    elif results[index].find('img').get("data-src") is not None:
                                        image_list = results[index].find('img').get('data-src')
                            #THE lINK BEGIN WITH HTTP OR HTTPS then keep it originally 
                            else:
                               
                                site = results[index].get('href')
                                desc = removespecial(results[index].text)
                                if results[index].get('title') is not None:
                                    title= removespecial(results[index].get('title'))
                                    
                                if results[index].find('div') is not None:
                                    photo = results[index].find(lambda tag: tag.name == 'div' and 'data-src' in tag.attrs)
                                    if photo is not None:
                                        image_list = photo.get("data-src")
                                if results[index].find('img') is not None:
                                    if results[index].find('img').get('src') is not None:
                                        image_list = results[index].find('img').get('src')
                                    elif results[index].find('img').get('data-img-src') is not None:
                                        image_list = results[index].find('img').get('data-img-src')
                                    elif results[index].find('img').get("data-src") is not None:
                                        image_list = results[index].find('img').get('data-src')
                                        
                            sites.append(results[index].get('href'))
                            #if image_list.startswith("http")==False:
                                #    image_list = url+image_list
                            found_results.append({"Title":title,"Link": site,'Description': desc,"Image":image_list,'Keyword': i})
                            title=""
                            desc=""
                            site=""
                            image_list=""
                        #Check if the link already in list, in case same keyword then do nothing else add new keyword to that link
                        else:
                            n = sites.index(results[index].get('href'))
                            if i in found_results[n]['Keyword']:
                                found_results[n]['Keyword'] = found_results[n]['Keyword']
                            else:
                                found_results[n]['Keyword'] = found_results[n]['Keyword']+"," + i
        
    except Exception as e:
        print(e)
        pass
    return found_results

#######
from nltk.corpus import wordnet as wn
def mainsearchwiththreads(dataframefile,searchstr):
    #print(removeduplicates(searchstr+" " + pathtostring(dataframefile)))
    #Merge all data from search categories
    df = pd.concat([pd.read_csv(open(f, 'r',encoding='utf8')) for f in dataframefile])
    #convert to list and remove duplicates
    urls=list(set(df["Link"].tolist()))
    searchstround = removeduplicates(searchstr+" " + pathtostring(dataframefile))
    searchwords = word_synonymtwo(searchstround)+wikipedia.search(searchstround,results=3)
    que = queue.Queue()
    prio_queue = queue.PriorityQueue() #Set priority queue for wiki search cos it take long to get info from wiki api
    threads_list = list()
    wn.ensure_loaded()
    wikire = Thread(target=lambda q, arg1: q.put(wiki(arg1)), args=(prio_queue,searchwords))
    wikire.daemon = True
    wikire.start()
    threads_list.append(wikire)
    for i in urls:
        t = Thread(target=lambda q, arg1,arg2: q.put(parselinkimg(arg1,arg2)), args=(que, i,searchwords))
        t.daemon = True
        t.start()
        threads_list.append(t)
    # Join all the threads
    for l in threads_list:
        l.join()

    # Check thread's return value
    found_results=[]
    while not que.empty():
            found_results.extend(que.get())
    wiki_item=[]
    while not prio_queue.empty():
        wiki_item.extend(prio_queue.get())

    df1 = pd.DataFrame(found_results,columns =['Title', 'Link', 'Description','Image','Keyword'])
    df2 = pd.DataFrame(wiki_item,columns =['Title', 'Link', 'Description','Image','Keyword','Ranking'])# for wikipedia search without ranking,always put on top
    if len(df1) ==0 and len(df2) ==0:
        return None
    if len(df1) > 0:
        df1 = rankindex(f"{searchstr}",df1,"findout").reset_index(drop=True)
        for index, row in df1.iterrows():
            df1.loc[[index],"Ranking"] =  int(countmatches(searchstr,row.Keyword))
        df1.sort_values('Ranking', ascending=False, inplace=True)
    
    dfresult=df2.append(df1).reset_index(drop=True)
    dfresult.to_csv("static//js//Category//findout.csv",index=False)
    data = dfresult.T.to_dict().values()
    print(data)
    if len(data)>0:
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        searchresults = db.searchresults
        db.drop_collection(searchresults)
        searchresults.insert_many(data)
    return dfresult.to_records().tolist()
####### without using Doc2Vec

def mainsearchwiththreads1(dataframefile,searchstr):
    #print(removeduplicates(searchstr+" " + pathtostring(dataframefile)))
    #Merge all data from search categories
    df = pd.concat([pd.read_csv(open(f, 'r',encoding='utf8')) for f in dataframefile])
    #convert to list and remove duplicates
    urls=list(set(df["Link"].tolist()))
    searchstround = removeduplicates(searchstr+" " + pathtostring(dataframefile))
    searchwords = word_synonymtwo(searchstround)+wikipedia.search(searchstround,results=3)
    que = queue.Queue()
    prio_queue = queue.PriorityQueue() #Set priority queue for wiki search cos it take long to get info from wiki api
    threads_list = list()
    wn.ensure_loaded()
    wikire = Thread(target=lambda q, arg1: q.put(wiki(arg1)), args=(prio_queue,searchwords))
    wikire.daemon = True
    wikire.start()
    threads_list.append(wikire)
    for i in urls:
        t = Thread(target=lambda q, arg1,arg2: q.put(parselinkimg(arg1,arg2)), args=(que, i,searchwords))
        t.daemon = True
        t.start()
        threads_list.append(t)
    # Join all the threads
    for l in threads_list:
        l.join()

    # Check thread's return value
    found_results=[]
    while not que.empty():
            found_results.extend(que.get())
    wiki_item=[]
    while not prio_queue.empty():
        wiki_item.extend(prio_queue.get())

    df1 = pd.DataFrame(found_results,columns =['Title', 'Link', 'Description','Image','Keyword'])
    df2 = pd.DataFrame(wiki_item,columns =['Title', 'Link', 'Description','Image','Keyword','Ranking'])# for wikipedia search without ranking,always put on top
    if len(df1) ==0 and len(df2) ==0:
        return None
    if len(df1) > 0:
        df1.reset_index(drop=True)
        for index, row in df1.iterrows():
            df1.loc[[index],"Ranking"] =  int(countmatches(searchstr,row.Keyword))
        df1.sort_values('Ranking', ascending=False, inplace=True)
    
    dfresult=df2.append(df1).reset_index(drop=True)
    dfresult.to_csv("static//js//Category//findout.csv",index=False)
    data = dfresult.T.to_dict().values()
    print(data)
    if len(data)>0:
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        searchresults = db.searchresults
        db.drop_collection(searchresults)
        searchresults.insert_many(data)
    return dfresult.to_records().tolist()
    
#######
#Train the search result by using those Description
def doctrain(documents,modelname):
    desc = (documents["Description"]+ documents["Keyword"]).tolist()
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(desc)]
    
    max_epochs = 15
    #if len(tagged_data) > max_epochs:
       # max_epochs=15
   # else:
      #  max_epochs=len(tagged_data)-1
   
    vec_size = 10
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1,dm_mean=1,max_vocab_size=None,workers=12)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save(f"d{modelname}v.model")
    print("Model Saved")
#again indexing(ranking) the results based on the model just trained and keywords/searchstring
def rankindex(searchstring,documents,modelname):
    doctrain(documents,modelname)
    model= Doc2Vec.load(f"d{modelname}v.model")
    #to find the vector of results which were in training data
    test_data = word_tokenize(searchstring.lower())
    #test_data = word_tokenize(" ".join(word_synonymall(searchstring)).lower())
    v1 = model.infer_vector(test_data)
    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar([v1], topn=len(documents))
    listindex = []
    for x in similar_doc:
        listindex.append(int(x[0]))
    return documents.iloc[listindex,:]
#Again manually ranking the results by count matches keywords with search string



 #Train the images search result by using those Description
def doctrainimages(documents,modelname):
    desc = documents["Description"].tolist()
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(desc)]
    max_epochs = 15
    vec_size = 10
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1,dm_mean=1,max_vocab_size=None,workers=12)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save(f"d{modelname}v.model")
    print("Model Saved")
#again indexing(ranking) the images results based on the model just trained and keywords/searchstring
def rankindeximages(searchstring,documents,modelname):
    doctrainimages(documents,modelname)
    model= Doc2Vec.load(f"d{modelname}v.model")
    #to find the vector of results which were in training data
    test_data = word_tokenize(searchstring.lower())
    #test_data = word_tokenize(" ".join(word_synonymall(searchstring)).lower())
    v1 = model.infer_vector(test_data)
    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar([v1], topn=len(documents))
    listindex = []
    for x in similar_doc:
        listindex.append(int(x[0]))
    return documents.iloc[listindex,:]    
   

#Again manually ranking the results by count matches keywords with search string
def countmatches(searchstr,cstring):    
    count = 0
    for x in cstring.split(","):
        #if len(x.split()) <=1:
        count = count + len(set(get_ngrammatches(searchstr.lower())).intersection(get_ngrammatches(x.lower())))
        #else:
            #count = count + len(set(get_ngrammatches(searchstr)).intersection(get_ngrams(x)))
    return count

def matchimage(searchstr,cstring):    
    count = 0
    for x in cstring.split():
        #if len(x.split()) <=1:
        count = count + len(set(get_ngrammatches(searchstr)).intersection(get_ngrammatches(x)))
        #else:
            #count = count + len(set(get_ngrammatches(searchstr)).intersection(get_ngrams(x)))
    return count



###After Process

def voice():
    file = "static//js//Category//findout.csv"
    state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
    state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128
    df = pd.read_csv(file)
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(str(len(df)) + "search results")
    a = win32api.GetKeyState(0x01)
    b = win32api.GetKeyState(0x02)  
    for x in range(len(df)):
        strvoice = df['Description'][x]
        speak.Speak(f"The link {x+1} {strvoice}")
        if state_left:  # Button state changed           
            driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
            driver.get(df['Link'][x])
#################################################
# Flask Setup
#################################################
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Query the database and send the jsonified results
@app.route("/receiver", methods=["GET", "POST"])
def jsdata():
    #data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
    if request.method == "POST":
        data = request.get_json()    
        print(data[0]['file'])
        speak.Speak("Searching")
        #speak.Speak(mainsearch(data[0]['file'],data[0]['searchstring']))
        #data1 = mainsearchwiththreads(data[0]['file'],data[0]['searchstring'])
        data1 = mainsearchwiththreads1(data[0]['file'],data[0]['searchstring'])
        #data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
        return jsonify(data1)
   
    #else:
      #  data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
      #  return jsonify(data1)
    #speak = wincl.Dispatch("SAPI.SpVoice")
   # speak.Speak(f"Search Results {df}")
    #speak.Speak(f"Search Completed")

@app.route("/data",methods = ["GET"])
def mlabdata():
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        
        db = client.cryptocurrency
        searchresults = db.searchresults.find({},{'_id':0})
        check =[]
        for document in searchresults:
            print(document)
            check.append(document)
        return jsonify(check)

@app.route("/voice", methods=["GET", "POST"])
def speech():
    if request.method == "POST":
        driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
        voice()
        return render_template("index.html")

##For image searching
@app.route("/receiverimage", methods=["GET", "POST"])
def jsdataimage():
    #data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
    if request.method == "POST":
        data = request.get_json()    
        print(data[0]['file'])
        speak.Speak("Searching")
        #speak.Speak(mainsearch(data[0]['file'],data[0]['searchstring']))
        data1 = mainsearchimages(data[0]['file'],data[0]['searchstring'])
        #data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
        return jsonify(data1)

@app.route("/imagedata",methods = ["GET"])
def mlabdataimage():
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        
        db = client.cryptocurrency
        searchresultsimages = db.searchresultsimages.find({},{'_id':0})
        img =[]
        for document in searchresultsimages:
            print(document)
            img.append(document)
        return jsonify(img)

@app.route("/analysisinput", methods=["GET", "POST"])
def analysisinput():
    #data1 = {'username': 'Pang', 'site': 'stackoverflow.com'}
    if request.method == "POST":
        data = request.get_json()
        checking = get_sentencenumber(data[0])    
        print(data)
        #data1 = mainsearchwiththreads(data[0]['file'],data[0]['searchstring'])
        print(checking)
        return jsonify(checking)

@app.route("/analysisdata",methods = ["GET"])
def mlabanalysis():
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        
        db = client.cryptocurrency
        searchanalysis = db.analyst.find({},{'_id':0})
        ana =[]
        for document in searchanalysis:
            print(document)
            ana.append(document)
        return jsonify(ana)

@app.route("/analysis", methods=["GET"])
def analysis():
    print("check.html")
    return render_template("check.html")   
#voice()  

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
if __name__ == "__main__":
    app.run(debug=True,threaded=True)