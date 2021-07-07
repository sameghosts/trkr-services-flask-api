from flask import Flask, request, json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob, Word
import string
import sys
#add rebeccas dependencies
#add a dummy json object
#try dummy import using posting
#add axios call in backend 

app = Flask(__name__)

# Add app.route methods = POST
@app.route("/")
def hello():
  return "Hello World from the Flask TRKR Service API"

#route for service
@app.route("/runNLP")
def nlp():
  return "Hello from NLP"

#app .run
if __name__ == "__main__":
  app.run()


