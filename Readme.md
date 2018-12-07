Search Capacity
![png](Capacity.png)
![png](Categories.png)

"Text Search"
![png](searchsample.png)
![png](searchsample1.png)
![png](searchsample2.png)

"Image Search"
![png](imagesr.png)

"Text and Number Analysis"
![png](analy.png)


```python
import nltk 
from nltk.corpus import wordnet 
synonyms = [] 
antonyms = [] 
for syn in wordnet.synsets("Tiger"): 
    for l in syn.lemmas(): 
        synonyms.append(l.name()) 
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name()) 
print(synonyms)
#print(set(antonyms)) 
```


```python
import nltk 
from nltk.corpus import wordnet 
```


```python
result = wordnet.synsets("good")
```


```python
print(result)
```


```python
import pronouncing
pronouncing.rhymes("climbing")
['diming', 'liming', 'priming', 'rhyming', 'timing']
```


```python
from pynput.mouse import Listener
import logging
from selenium import webdriver
from pynput import keyboard
import time
```


```python
logging.basicConfig(filename="log.txt",level=logging.DEBUG,format = "%(asctime)s: %(message)s")
#def on_move(x,y):
state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128

break_program = False
def on_click(x,y,button,pressed):
    #logging.info("Mouse clicked at {0},{1} with {2}".format(x,y,button))
    #driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
    #driver.get("http://www.google.com")
    #global break_program
    if pressed == "Button.left":
        print ('Open the link and end the program')
        break_program = True
        #return False
#def on_scroll(x,y,dx,dy)
with Listener(on_click=on_click) as listener:
    while break_program == False:
        print ('Program running')
        time.sleep(5)
    listener.join()

#with Listener (on_click = on_click) as listener:
   # listener.join()
    
```


```python
import win32api
import time
from selenium import webdriver

state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128
break_program = False
while break_program == False:
    a = win32api.GetKeyState(0x01)
    b = win32api.GetKeyState(0x02)

    if a != state_left:  # Button state changed
        state_left = a
        print(a)
        print('Left Button Pressed')
        driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
        driver.get("http://www.google.com")
        #
        #if a < 0:
            
        #else:
           # print('Left Button Released')

    if b != state_right:  # Button state changed
        state_right = b
        print(b)
        #if b < 0:
        print('Right Button Pressed')
        break_program = True
        #else:
         #   print('Right Button Released')
    time.sleep(0.5)
```


```python
import nltk
from nltk.corpus import PlaintextCorpusReader

#corp = PlaintextCorpusReader(r'C:\Users\user\Desktop\Project 3','log.txt')
text="Right Mouse with then right mouse with left mouse With"
#text = nltk.Text(corp.words())
print(text.lower().count("mouse"))
string = "Right Mouse with"
match = [word for word in string.split()]
```


```python
match
text.dispersion_plot(match)

```


```python
for a in match:
    b= text.concordance(a)
    print(b)
```


```python
sent = 'Rose is a rose is a rose is a rose.'
toks = nltk.word_tokenize(sent)
print(toks)

```


```python
from textblob import Word
import sys
import random

lines = "Hello great nice good career"
#for line in lines:
    #line = line.strip()
    #line = line.decode('ascii', errors="replace")
    #words = line.split()
    #output = list()
words = lines.split()
#print(words)
for word_str in words:
    word_obj = Word(word_str)
    #print(word_obj)
   # print(word_str)
   
    synonyms = [] 

    for syn in wordnet.synsets(word_obj): 
        for l in syn.lemmas():
            if l.name() != word_obj:
                synonyms.append(l.name())
                print(l.name())
    print(random.choice(synonyms))
    print("##########new word############")
    #random_lemma = random.choice(random_synset.lemmas())
    #print(random_synset.lemmas())
    #if len(word_str) > 3 and len(word_obj.synsets) > 0:
      #  random_synset = random.choice(word_obj.synsets)
       # print(random_synset)
        #random_lemma = random.choice(random_synset.lemmas())
        #output.append(random_lemma.replace('_', ' '))
    #else:
        #output.append(word_str)
    #print(output)
```


```python
!pip install gensim
```


```python
!pip install defaultdict
```


```python
# Tokenize Corpus and filter out anything that is a
# stop word or has a frequency <1

from gensim import corpora, models, similarities
from collections import defaultdict

documents = [
    'Tiger Woods and the case for him making the 2018 United States Jul 31, 2018 - Tiger Woods is a long shot to move into the top eight in the next two events and guarantee himself a spot ... But pretty much a lock',  # doc_id 7
]

stoplist = set(['is', 'how'])

texts = [[word.lower() for word in document.split()
          if word.lower() not in stoplist]
         for document in documents]

print(texts)
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts)

# doc2bow counts the number of occurences of each distinct word,
# converts the word to its integer word id and returns the result
# as a sparse vector

corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = "How many tigers in US"
vec_bow = dictionary.doc2bow(doc.lower().split())

# convert the query to LSI space
vec_lsi = lsi[vec_bow]
index = similarities.MatrixSimilarity(lsi[corpus])

# perform a similarity query against the corpus
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print(sims)
```


```python
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
```


```python
data = ["for the states of Andhra Pradesh","Odisha and Jharkhand","most of the tiger-range states India now hosts 70 per cent of the world tiger population which has",
        "This is a very healthy and natural rate for any carnivore population to grow in a"]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate()]
print(tagged_data)

```


```python
check = ["for the states of Andhra Pradesh","Odisha and Jharkhand","most of the tiger-range states India now hosts 70 per cent of the world tiger population which has",
        "This is a very healthy and natural rate for any carnivore population to grow in a"]
```


```python
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(check)]
print(tagged_data)
```


```python
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
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

model.save("d2v.model")
print("Model Saved")
```


```python
from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("natural".lower())
#print(test_data)
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar([v1], topn=20)
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])
```


```python
listindex = []
for x in similar_doc:
    listindex.append(int(x[0]))
```


```python
listindex
```


```python
df1= df.iloc[listindex,:]
```


```python
df1.reset_index()
```


```python
df1.to_csv("check.csv")
```


```python
!pip install wikipedia
```


```python
#Check strings with words are nouns. Set priority with Uppercase then first letter upper then lowercase for searching
import nltk
lines = 'trump is a city of vietnam'
# function to test if something is a noun
is_noun = lambda pos: pos[:2] == 'NN'
# do the nlp stuff
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
search = [word for word in nouns if not word.islower() and not word.isupper()] + [word for word in nouns if word.isupper()]
if len(search)==0:
    search = [word for word in nouns if word.islower()]
search
```


```python
import wikipedia

def wiki(searchstr):
    results = wikipedia.search(searchstr,results=3)
    print(results)
    found_results=[]
    title=""
    url=""
    content=""
    image=[]
    ranking=2
    if len(results) >0:
        for x in results:
            try:
                result = wikipedia.page(x)
                if result.title != "":
                    title= result.title
                if result.url != "":
                    url = result.url
                print(url)
                if wikipedia.summary(x,sentences=2) != "":
                    content =  wikipedia.summary(x,sentences=2)
                print(len(result.images))
                if len(result.images) >0:
                    image = result.images[0]
                found_results.append({"Title":title,"Link":url,'Description':content,"Image":image,'Keyword':title,'Ranking':ranking})
            except wikipedia.exceptions.DisambiguationError as e:
                print(e)
                pass
            except KeyError:
                pass
    return found_results

#ny = wikipedia.page("DD")
#ny.content
```


```python
wikipedia.search("vietnam",results=3)
```


```python
def removeduplicates(string):
    words = string.split()
    return " ".join(sorted(set(words), key=words.index))
```


```python
wikipedia.search(removeduplicates("trump breaking news news"),results=3)
```


```python
wiki("who is boxin")
```


```python

import pandas as pd

#df=pd.DataFrame(check,columns =['Title', 'Link', 'Description','Image','Keyword','Ranking'])
prio_queue = queue.PriorityQueue() #Set priority queue for wiki search cos it take long to get info from wiki api
threads_list = list()
wikire = Thread(target=lambda q, arg1: q.put(wiki(arg1)), args=(prio_queue,"cupps"))
wikire.start()
threads_list.append(wikire)
wiki_item=[]
for t in threads_list:
        t.join()
while not prio_queue.empty():
    wiki_item.extend(prio_queue.get())
    print("halloooo")
print(wiki_item)
```


```python

from bs4 import BeautifulSoup
import requests
url = "http://en.wikipedia.org/wiki/sanjose"

soup = BeautifulSoup(requests.get(url).content)
soup
```


```python
documents = ["for the states of Andhra Pradesh","Odisha and Jharkhand","most of the tiger-range states India now hosts 70 per cent of the world tiger population which has",
        "This is a very healthy and natural rate for any carnivore population to grow in a"]
searchstring = "population"

doctrain(documents)
print(rankindex(searchstring,documents))


```


```python
def get_digits(str1):
  
    for i in str1:
        if i.isdigit():
            c = c + i
    return c  
```


```python

    
#listOfNumbers = [float(i) for i in newstr.split()]
#print(listOfNumbers)
```


```python
get_number("M, check 10,000 then , 50.000 then 10 then 5")
```


```python
#Select sentence with numbers:
import re
samplereport="Senate to more than 32,000 votes Sunday as the state's gradual count continued. I image from 08/25 through 12. The patient image 1.2, 23, 34, 45 and 64 from serise 34. image look good to have a tumor in this area.  It has been resected during the interval between scans.  The"
#pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
sentence=re.findall('[^.!?]+[.!?]',samplereport)
#a = sentence.findall(SampleReport)
for x in sentence:
    print(get_digits(x))
    
```


```python
!pip install inscriptis

```


```python
import urllib.request
import re
from inscriptis import get_text
import pandas as pd
import pymongo

def get_number(s):
    newstr = ''.join((ch if ch in '0123456789.,/:%$' else ' ') for ch in s)
    return newstr

def get_sentencenumber(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html)
    title = soup.find('h1').text
    print(title)
    text = get_text(html)
    text.replace('"',"" )
    #result1 = re.sub(r"\s+"," ", text, flags = re.I)
    #sentence=re.compile("[A-Z].*?[\.!?;] ", re.MULTILINE | re.DOTALL )
    a =   re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
   # a = sentence.findall(result1)
    content = []
    number=[]
    #pre=""
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
            #pre=""
        #else:
            #pre=x
    
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
    
```


```python
!pip install flask-cors
```


```python
get_sentencenumber("http://www.cbsnews.com/news/trump-threatens-french-wine-in-tuesday-tweetstorm/")
```


```python
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

html = urllib.request.urlopen('https://thanhnien.vn/thoi-su/cuu-trung-tuong-phan-van-vinh-de-nghi-khong-cong-bo-ban-an-len-trang-toa-an-1022546.html').read()

text = text_from_html(html)
result1 = re.sub(r"\s+"," ", text, flags = re.I)
#print(text)

sentence=re.compile("[A-Z].*?[\.!?] ", re.MULTILINE | re.DOTALL )
a = sentence.findall(result1)
for x in a:
    result = re.findall(r"\d+", x)
    if len(result) >0:
        print(x)
        print("The Numbers:" + re.sub(r"\s+"," ",get_number(x), flags = re.I))

```


```python
import requests
import win32com.client as wincl
from bs4 import BeautifulSoup
import win32api
import time
from selenium import webdriver
from splinter import Browser
import re

def searchcheck():
        USER_AGENT = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
        url = 'https://www.nytimes.com/'
        response = requests.get(url, headers=USER_AGENT)
        response.raise_for_status()
        # Parsing
        #executable_path = {"executable_path": "chromedriver"}
        #browser = Browser("chrome", **executable_path, headless=False)
        # random sleep for 3-10 sec
        #url = "https://www.google.com/search?q=hiking&num=200&hl=en"
        #browser.visit(url)
        #html = browser.html
        #soup = BeautifulSoup(html, "html.parser")
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)
        found_results = []
        #for element in soup.find_all(text=re.compile("technology")):
          #  found_results.append(element.get_text())

        return found_results
```


```python
searchcheck()
```


```python
!pip install Scrapy

```


```python
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class MySpider(CrawlSpider):
    name = 'crawlspider'
    start_urls = ['http://scrapy.org']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(response):
        item = dict()
        item['url'] = response.url
        item['title'] = response.meta['link_text']
        # extracting basic body
        item['body'] = '\n'.join(response.xpath('//text()').extract())
        # or better just save whole source
        item['source'] = response.body
        print(item)
        return item
```


```python
 MySpider.item
```


```python
import requests
from scrapy.http import TextResponse

res = requests.get('https://www.cnn.com')
response = TextResponse(res.url, body=res.text, encoding='utf-8')
```


```python
print('\n'.join(response.xpath('//text()').extract()))
```


```python
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec

#Train the search result by using those Description and Keyword
def doctrain(documents,modelname):
    desc = (documents["Description"]+ documents["Keyword"]).tolist()
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
    #results = []
    #for i in listindex:
     #   results.append(documents[i])
    #return results
    

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
    counts = 0
    for x in cstring.split():
        #if len(x.split()) <=1:
        counts = counts + len(set(get_ngrammatches(searchstr)).intersection(get_ngrammatches(x)))
        #else:
            #count = count + len(set(get_ngrammatches(searchstr)).intersection(get_ngrams(x)))
    print("checking" + str(counts))
    return counts
    
    
```


```python
matchimage("U.S VOTE","votes")
```


```python
!pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
```


```python
from imageai.Prediction import ImagePrediction
import os

```


```python
execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel(prediction_speed="fastest")

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "Images//husky.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
```


```python
# Dependencies
import matplotlib.pyplot as plt
%matplotlib inline

import os
import numpy as np
import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
```


```python
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

```


```python
image_path = os.path.join("https://static01.nyt.com/images/2018/10/28/sports/28isaiah-web-washington/28isaiah-web-washington-jumbo.jpg")
a =predict(image_path)
a[0][1]
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-6-aca399d07516> in <module>()
          1 image_path = os.path.join("https://static01.nyt.com/images/2018/10/28/sports/28isaiah-web-washington/28isaiah-web-washington-jumbo.jpg")
    ----> 2 a =predict(image_path)
          3 a[0][1]
    

    <ipython-input-4-4364fdd0a6b0> in predict(image_path)
          5     weights='imagenet')
          6     image_size = (299, 299)
    ----> 7     img = image.load_img(image_path, target_size=image_size)
          8     x = image.img_to_array(img)
          9     x = np.expand_dims(x, axis=0)
    

    ~\Anaconda3\lib\site-packages\keras_preprocessing\image.py in load_img(path, grayscale, color_mode, target_size, interpolation)
        493         raise ImportError('Could not import PIL.Image. '
        494                           'The use of `array_to_img` requires PIL.')
    --> 495     img = pil_image.open(path)
        496     if color_mode == 'grayscale':
        497         if img.mode != 'L':
    

    ~\Anaconda3\lib\site-packages\PIL\Image.py in open(fp, mode)
       2546 
       2547     if filename:
    -> 2548         fp = builtins.open(filename, "rb")
       2549         exclusive_fp = True
       2550 
    

    OSError: [Errno 22] Invalid argument: 'https://static01.nyt.com/images/2018/10/28/sports/28isaiah-web-washington/28isaiah-web-washington-jumbo.jpg'



```python
from textblob import Word
import sys
import random
import nltk 
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
from textblob import Word

# get searchstring by using N-gram to get combined words from 2 - i (i less than total length of search string) words in search string
def get_ngrams(text):
    total =[]
    n=2
    while n <=len(word_tokenize(text)):
        n_grams = ngrams(word_tokenize(text.lower()), n)
        n = n+1
        total = total + [' '.join(grams) for grams in n_grams]
    return total
#Get Ngram with lemmatize to singular and convert Verb to original before compare to get right indexes for search results
def get_ngrammatches(text):
    a=list(set(word_tokenize(text.lower())+ [Word(wordlemma).lemma for wordlemma in word_tokenize(text.lower())]+ [Word(wordlemma).lemmatize("v") for wordlemma in word_tokenize(text.lower())]))
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

# Find 2 similar meaning of each word in searching string plus search string, equal len(searchstring)x3+1 with N-gram and remove stop words
def word_synonymtwo(lines):
    try:
        df =pd.read_csv("stopword.csv")
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



```


```python
get_ngrammatches("checking")
```


```python
wordnet.synsets("ci")
```


```python
word_synonymtwo("  dasd  sad    ")
```


```python
import pandas as pd

documents = 'Tiger Woods and the case for him making the 2018 United States Jul 31, 2018 - Tiger Woods is a long shot to move into the top eight in the next two events and guarantee himself a spot ... But pretty much a lock'  # doc_id 7

#stoplist = set(
#])
df =pd.read_csv("stopword.csv")
stopwords=df["stopwords"].tolist()

texts = [word.lower() for word in documents.split()
          if word.lower() not in stopwords]
         

print(' '.join(texts))

```


```python
import re
from nltk.util import ngrams

def check(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, 2))
    return output
```


```python
get_ngrams("Natural language use")
```


```python
#get_ngrams ("Natural language processing vietnam chxnch dhh")
word_synonymtwo("Natural world in u.s")

```


```python

synonymall("Natural language")
```


```python
word_synonymall("Natural language processing")

```


```python
word_synonymtwo("hallo chao boi sang ")
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd

df =pd.read_csv("check.csv")
urls=df["Link"].tolist()
found_results = []
site=""
desc=""
title=""
sites=[]
#url = ["https://www.npr.org/sections/news","https://www.bbc.com","https://www.cnn.com","http://www.foxnews.com","https://www.vpdinh.com"]
for url in urls:
    print(url)
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        ss = "Los Angeles"
        #print(word_synonymtwo(ss))
        for i in word_synonymtwo(ss):
           #print(i)
            links = lambda tag: getattr(tag, 'name', None) == 'a' and 'href' in tag.attrs and i.lower() in tag.get_text().lower()
            results = soup.find_all(links)
            print(len(results))
            if len(results) > 0:
                for index in range(len(results)):
                    #print(results[index])
                    if sites.count(results[index].get('href'))==0:
                        if (results[index].get('href')[0:4].lower() != "http" and results[index].get('href')[0:3].lower() != "www" and results[index].get('href')[0:4].lower() != "//ww" and results[index].get('href')[0:1]!="."):
                            site = url+results[index].get('href')
                            desc = removespecial(results[index].text)
                            title= removespecial(results[index].get('title'))
                        elif (results[index].get('href')[0:1]=="."):
                            site = url+results[index].get('href')[1:]
                            desc = removespecial(results[index].text)
                            title= removespecial(results[index].get('title'))
                        else:
                            site = results[index].get('href')
                            desc = removespecial(results[index].text)
                            title=  removespecial(results[index].get('title'))
                        sites.append(results[index].get('href'))        
                        found_results.append({"Title":title,"Link": site,'Description': desc,'Keyword': i})
                        title=""
                        desc=""
                        site=""
    except Exception as e:
        print("Not Found")
        
```


```python
len(found_results)
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd
import pymongo

def mainsearch(dataframefile,searchstr):
    df =pd.read_csv(dataframefile)
    urls=df["Link"].tolist()
    found_results = []
    site=""
    desc=""
    title=""
    sites=[]
    for url in urls:     
        try:
            r = requests.get(url,timeout=3)
            print(r.status_code)
            soup = BeautifulSoup(r.content)
            for i in word_synonymtwo(searchstr):
                print(i)
                links = lambda tag: getattr(tag, 'name', None) == 'a' and 'href' in tag.attrs and i.lower() in tag.get_text().lower()
                results = soup.find_all(links)
                print(len(results))
                if len(results) > 0:
                    for index in range(len(results)):
                        #print(results[index])
                        if sites.count(results[index].get('href'))==0:
                            if (results[index].get('href')[0:4].lower() != "http" and results[index].get('href')[0:3].lower() != "www" and results[index].get('href')[0:4].lower() != "//ww" and results[index].get('href')[0:1]!="."):
                                site = url+results[index].get('href')
                                desc = removespecial(results[index].text)
                                title= removespecial(results[index].get('title'))
                            elif (results[index].get('href')[0:1]=="."):
                                site = url+results[index].get('href')[1:]
                                desc = removespecial(results[index].text)
                                title= removespecial(results[index].get('title'))
                            else:
                                site = results[index].get('href')
                                desc = removespecial(results[index].text)
                                title=  removespecial(results[index].get('title'))
                            sites.append(results[index].get('href'))        
                            found_results.append({"Title":title,"Link": site,'Description': desc,'Keyword': i})
                            title=""
                            desc=""
                            site=""
        except Exception as e:
            print("Not Found Any Info")
    df1 = pd.DataFrame(found_results,columns =['Title', 'Link', 'Description','Keyword'])
    print(df1)
    if len(df1) > 0:
        dffinal = rankindex(f"searchstr",df1,"findout").reset_index(drop=True)
        dffinal.to_csv("static//js//Category//findout.csv",index=False)
        data = dffinal.T.to_dict().values()
        print(data)
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        searchresults = db.searchresults
        db.drop_collection(searchresults)
        searchresults.insert_many(data)
        return f"{len(dffinal)} Search Results"
    else:
        return "Search Not Found"
   
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd

def mainsearchone(dataframefile,searchstr):
    df =pd.read_csv(dataframefile)
    urls=df["Link"].tolist()
    found_results = []
    site=""
    desc=""
    title=""
    sites=[]
    for url in urls:     
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content)
            for i in word_synonymone(searchstr):
                print(i)
                links = lambda tag: getattr(tag, 'name', None) == 'a' and 'href' in tag.attrs and i.lower() in tag.get_text().lower()
                results = soup.find_all(links)
                print(len(results))
                if len(results) > 0:
                    #print(results[index])
                    if sites.count(results[0].get('href'))==0:
                        if (results[0].get('href')[0:4].lower() != "http" and results[0].get('href')[0:3].lower() != "www" and results[0].get('href')[0:4].lower() != "//ww" and results[0].get('href')[0:1]!="."):
                            site = url+results[0].get('href')
                            desc = removespecial(results[0].text)
                            title= removespecial(results[0].get('title'))
                        elif (results[0].get('href')[0:1]=="."):
                            site = url+results[0].get('href')[1:]
                            desc = removespecial(results[0].text)
                            title= removespecial(results[0].get('title'))
                        else:
                            site = results[0].get('href')
                            desc = removespecial(results[0].text)
                            title=  removespecial(results[0].get('title'))
                        sites.append(results[0].get('href'))        
                        found_results.append({"Title":title,"Link": site,'Description': desc,'Keyword': i})
                        title=""
                        desc=""
                        site=""
        except Exception as e:
            print("Not Found Any Info")
    df1 = pd.DataFrame(found_results,columns =['Title', 'Link', 'Description','Keyword'])
    if len(df1) > 0:,
        dffinal = rankindex(f"searchstr",df1,"findout").reset_index(drop=True)
        dffinal.to_csv("static//js//Category//findout.csv",index=False)
        return dffinal
    else:
        return "Search Not Found"
   
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd

def parselink(url,searchstr):
    found_results = []
    site=""
    desc=""
    title=""
    sites=[]
    try:
        r = requests.get(url,timeout=3)
        if r.status_code  == 200:
            soup = BeautifulSoup(r.content,"lxml")
            for i in word_synonymtwo(searchstr):
                print(i)
                links = lambda tag: getattr(tag, 'name', None) == 'a' and 'href' in tag.attrs and i.lower() in tag.get_text().lower()
                results = soup.find_all(links)
                print(len(results))
                if len(results) > 0:
                    for index in range(len(results)):
                        #print(results[index])
                        if sites.count(results[index].get('href'))==0:
                            if (results[index].get('href')[0:4].lower() != "http" and results[index].get('href')[0:3].lower() != "www" and results[index].get('href')[0:4].lower() != "//ww" and results[index].get('href')[0:1]!="."):
                                site = url+results[index].get('href')
                                desc = removespecial(results[index].text)
                                title= removespecial(results[index].get('title'))
                            elif (results[index].get('href')[0:1]=="."):
                                site = url+results[index].get('href')[1:]
                                desc = removespecial(results[index].text)
                                title= removespecial(results[index].get('title'))
                            else:
                                site = results[index].get('href')
                                desc = removespecial(results[index].text)
                                title=  removespecial(results[index].get('title'))
                            sites.append(results[index].get('href'))        
                            found_results.append({"Title":title,"Link": site,'Description': desc,'Keyword': i})
                            title=""
                            desc=""
                            site=""
                        else:
                            found_results[index]['Keyword'] = found_results[index]['Keyword']+"," + i
    except Exception as e:
        print("Not Found Any Info")
        pass
    return found_results

df =pd.read_csv("check.csv")
urls=df["Link"].tolist()
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd

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
       
        print(e)
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
```


```python
def checkparseimg(url,searchstr, soup):
    resultsalt = []
    desc=""
    title=""
    sites=[]
    image_list=""
    linklist=[]
    try:
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
     
        print(e)
        pass
    result = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in found_result2)] #Remove duplicate elements in list results
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
        
    return result
```


```python
#parseimg("http://www.cnbc.com",['trump out', 'trump', 'Donald Trump', 'Trump family', 'Melania Trump'])
a = ['trump out', 'trump', 'Donald Trump', 'Trump family', 'Melania Trump']
b= " ".join(k for k in a)
parseimg("http://www.cnbc.com",b)
```


```python
from bs4 import BeautifulSoup
import requests
from pprint import pprint
import pandas as pd

def parselinkimg(url,searchstr):
    found_results = []
    site=""
    desc=""
    title=""
    sites=[]
    image_list=""
    found_result2=[]
    try:
        r = requests.get(url,timeout=3)
        if r.status_code  == 200:
            #print("main check")
            soup = BeautifulSoup(r.content,"lxml")
            
            #print("main check1")
            #print(word_synonymtwo(searchstr))
            for i in searchstr:
                print(i)
                #found_result2.extend(checkparseimg(url,searchstr, soup))
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
                            if image_list.startswith("http")==False:
                                    image_list = url+image_list
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
        print("check error")
        print(e)
        #speak = wincl.Dispatch("SAPI.SpVoice")
        #speak.Speak()
        pass

df =pd.read_csv("news.csv")
urls=df["Link"].tolist()
```


```python
parselinkimg("http://www.cnbc.com","trump")
```


```python

from bs4 import BeautifulSoup
import requests
url = "https://en.wikipedia.org/wiki/orange county"
r = requests.get(url,timeout=3)
#if r.status_code  == 200:
soup = BeautifulSoup(r.content,"lxml")
soup.find('div',"mw-content-ltr").text
```


```python
result
```


```python
result[0]['Keyword'] = result[0]['Keyword']+"," + "hakkko"
result
```


```python
print(urls)
```


```python
!pip install queue
```


```python
import threading
import queue

def worker():
    """thread worker function"""
    print('Worker')
    return
threads =[]
#urls = ['https://www.theguardian.com/world', 'https://in.reuters.com/news/world', 'https://www.washingtonpost.com/world'] 
for i in range(len(urls)):
    t = threading.Thread(target=parselink, args=(urls[i],"new"))
    threads.append(t)
    t.start()

```


```python
mainsearch("check.csv","trump")
```


```python
import win32com.client as wincl
from bs4 import BeautifulSoup
import win32api

speak = wincl.Dispatch("SAPI.SpVoice")
speak.Speak(mainsearch("check.csv","today"))

```


```python
import queue
from threading import Thread

def foo(bar):
    print('hello {bar}')
    return bar

que = queue.Queue()
threads_list = list()

for i in urls:
    t = Thread(target=lambda q, arg1,arg2: q.put(parselink(arg1,arg2)), args=(que, i,"mexico"))
    t.start()
    threads_list.append(t)

# Join all the threads
for t in threads_list:
    t.join()

# Check thread's return value
result=[]
while not que.empty():
        result.extend(que.get())
print(result)
```


```python
import queue
from threading import Thread
import pymongo
from nltk.corpus import wordnet as wn

#Use multithreading to increase speed of the search engine.if python 2 can use multiprocessing
def mainsearchwiththreads(dataframefile,searchstr):
    print(dataframefile)
    df =pd.read_csv(dataframefile)
    urls=df["Link"].tolist()
    searchwords = word_synonymtwo(searchstr)+wikipedia.search(searchstr,results=3)
    que = queue.Queue()
    queimg = queue.Queue()
    prio_queue = queue.PriorityQueue() #Set priority queue for wiki search cos it take long to get info from wiki api
    threads_list = list()
    wn.ensure_loaded()
    wikire = Thread(target=lambda q, arg1: q.put(wiki(arg1)), args=(prio_queue,searchwords))
    wikire.daemon = True 
    wikire.start()
    threads_list.append(wikire)
    for i in urls:
        img = Thread(target=lambda q, arg1,arg2: q.put(parseimg(arg1,arg2)), args=(queimg, i,searchwords))
        img.daemon = True 
        img.start()
        threads_list.append(img)
        
        t = Thread(target=lambda q, arg1,arg2: q.put(parselinkimg(arg1,arg2)), args=(que, i,searchwords))
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
    wiki_item=[]
    while not prio_queue.empty():
        wiki_item.extend(prio_queue.get())
    
    image_item=[]
    while not queimg.empty():
        image_item.extend(queimg.get())
    
    print("checking images")
    print(image_item)
    df1 = pd.DataFrame(found_results,columns =['Title', 'Link', 'Description','Image','Keyword'])
    df2 = pd.DataFrame(wiki_item,columns =['Title', 'Link', 'Description','Image','Keyword'])
    if len(df1) > 0:
        dffinal = rankindex(searchstr,df1,"findout").reset_index(drop=True)
        for index, row in dffinal.iterrows():
            dffinal.loc[[index],"Ranking"] =  int(countmatches(searchstr,row.Keyword))
        dffinal.sort_values('Ranking', ascending=False, inplace=True)
        dffinal.to_csv("static//js//Category//findout.csv",index=False)
        dfresult = pd.concat([df2, dffinal], ignore_index=True)
        data = dfresult.T.to_dict().values()
        dbuser = "admin"
        dbpassword = "dinhdinh1981"
        conn=f"mongodb://{dbuser}:{dbpassword}@ds245082.mlab.com:45082/cryptocurrency"
        client = pymongo.MongoClient(conn)
        db = client.cryptocurrency
        searchresults = db.searchresults
        db.drop_collection(searchresults)
        searchresults.insert_many(data)
        return dfresult.to_records().tolist()
    else:
        return None
```


```python
import queue
from threading import Thread
import pymongo
from nltk.corpus import wordnet as wn

#Use multithreading to increase speed of the search engine.if python 2 can use multiprocessing
def mainsearchimages(dataframefile,searchstr):
    print(dataframefile)
    df =pd.read_csv(dataframefile)
    urls=df["Link"].tolist()
    que = queue.Queue()
    threads_list = list()
    wn.ensure_loaded()
    for i in urls:
        t = Thread(target=lambda q, arg1,arg2: q.put(parseimg(arg1,arg2)), args=(que, i,searchstr))
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
```


```python
for index, row in dffinal.iterrows():
    print("check key" + row['Keyword'])
```


```python
get_ngrammatches("hello")
```


```python
str1 = "hello in cair"
str1.count("aair")

from textblob import Word

text = "hello from pythons cars in california went"
a=list(set(word_tokenize(text)+ [Word(wordlemma).lemma for wordlemma in word_tokenize(text)]+ [Word(wordlemma).lemmatize("v") for wordlemma in word_tokenize(text)]))
print(a)
total =[]
n=1
while n <=len(word_tokenize(text)):
    n_grams = ngrams(a, n)
    n = n+1
    total = total + [' '.join(grams) for grams in n_grams]

#w = Word("men")
#w.lemmatize()
#'octopus'
#k= Word("thing")
#k.lemma

```


```python
def countmatches(searchstr,cstring):    
    counts = 0
    for x in cstring.split(","):
        #if len(x.split()) <=1:
        counts = counts + len(set(get_ngrammatches(searchstr)).intersection(get_ngrammatches(x)))
        
        #else:
            #count = count + len(set(get_ngrammatches(searchstr)).intersection(get_ngrams(x)))
    print("checking" + counts)
    return counts
```


```python
get_ngrammatches("trump")
```


```python
countmatches("automotile hybrid cars went","car,Hybrid,AutoMotile Hybrid go")
```


```python
word_synonymtwo("trump in birth citizen")
```


```python
pathtostring(["Category//sports//Table Tennis.csv","Category//sports//Tennis.csv"])
```


```python
mainsearchwiththreads("check.csv","trump china russian")
```


```python
mainsearchimages("check.csv",'U.S votes')
```


```python
get_ngrams("chao buoi sang Hello")
```


```python
import win32com.client as wincl
from bs4 import BeautifulSoup
import win32api
from selenium import webdriver
from splinter import Browser

def voice():
    file = "static//js//Category//findout.csv"
    state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
    state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128
    df = pd.read_csv(file)
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(str(len(df)) + "search results")
    a = win32api.GetKeyState(0x01)
    b = win32api.GetKeyState(0x02)
    print(a)
    print(b)
    for x in range(len(df)):
        strvoice = df['Description'][x]
        speak.Speak(f"The link {x+1} {strvoice}")
        if state_left:  # Button state changed           
            driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
            driver.get(df['Link'][x])

                #else:
                 #   print('Right Button Released')
      
   
```


```python
voice()
```


```python
dffinal.head()
```


```python
a = df["description"].tolist()
a
```


```python
doctrain(df["description"].tolist(),"check")
print(rankindex("Gas increase",df["description"].tolist(),"check"))
```


```python
a = [{"link":4,"num":2},{"link":3,"num":1},{"link":5,"num":6}]
b=a["link"]
b
```

