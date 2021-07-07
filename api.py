from flask import Flask, request, json
import pandas as pd
import numpy as np
import re
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
  #Function name: top_words
  # Passed: X_vect and vectorizer
  # Returns: top 5 occuring terms as DF
  def top_words(X, vect):
    temp_df = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
    temp_df = temp_df.sum().sort_values(ascending=False).head(5)
    temp_df = temp_df.reset_index()
    return (temp_df['index']) 

  # Function name: get_unique
  # Passed: column_name string and DF name
  # Returns: set of unique words found in column
  def get_unique(column_name, df):
    words = df[column_name].str.lower().str.findall("\w+")
    unique_words = set()
    for x in words:
      unique_words.update(x)
    return unique_words
  # Function name: create_stopwords
  # Passed: DF name
  # Returns: list collection of custom stopwords
  def create_stopwords(df):
    custom_stopwords = list(STOPWORDS)    # initialize base stopwords from wordcloud
    words_with_punc = [word for word in STOPWORDS if not all(ch not in word for ch in string.punctuation)]    # isolate base stopwords that contain punctuation
    stripped_wwp = [re.sub('[^a-zA-Z]','', c) for c in words_with_punc]    # strip punctuation from words_with_punc for safety (covers stopword stripping both pre/post nlp preprocessing)
    # expand stopwords to include common keywords from equal opportunity employment disclaimer
    EOE_words = [
      'equal', 
      'opportunity', 
      'sexual', 
      'orientation', 
      'race', 
      'gender', 
      'martial', 
      'status', 
      'color', 
      'religion', 
      'disability',
      'veteran',
      'origin',
      'identity']
    [custom_stopwords.append(word) for word in stripped_wwp]    # add stripped_wwp to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique('title', df)]    # add unique title words to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique('location', df)]    # add unique location words to custom_stopwords
    [custom_stopwords.append(word) for word in get_unique('company_name', df)]    # add unique company words to custom_stopwords
    [custom_stopwords.append(word) for word in EOE_words]    # add EOE disclaimer words to custom_stopwords
    return custom_stopwords
  # Function name: my_preprocessor
  # Passed: text
  # Returns: text as lowercase, stripped of non-alpha characters and extra spaces
  def my_preprocessor(text):
    text = [word for word in text.lower().split() if word not in string.punctuation]    #POSSIBLY REDUNDANT LINE
    text = [re.sub('[^a-zA-Z]','', c) for c in text]  # remove none alpha character
    text = [word for word in text if len(word)>0 if word not in custom_stopwords]     # remove empty character elements from list
    return ' '.join(text)
  
  # Function name: lemmatize_with_postag
  # Passed: sentence
  # Returns: lemmatized text, where the choices made be TextBlob lemmatizer are based on POS tag 
  def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {
      "J": 'a', 
      "N": 'n', 
      "V": 'v', 
      "R": 'r'
    }
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return lemmatized_list
  
  def run_comparison (n1, n2):
    custom_tf = TfidfVectorizer(preprocessor=my_preprocessor, tokenizer=lemmatize_with_postag, max_features = 500, ngram_range = (n1, n2))
    data_custom_tf = custom_tf.fit_transform(data)
    custom_tf_df = pd.DataFrame(data_custom_tf.toarray(), columns = custom_tf.get_feature_names())
    return top_words(data_custom_tf,custom_tf)

  df = pd.read_json(r'~/Users/seancemichael/Desktop/product_club/python-flask/trkr-SnS-flaskapi/right_details_jobs.json')

  custom_stopwords = create_stopwords(df)

  data = df['description']

  results_uni = run_comparison(1,1)
  results_bi = run_comparison(2,2)
  results_tri = run_comparison(3,3)

  results = {
      "num_jobs_compared": df.shape[0],
      "job_id_list": df['job_id'],
      "results_uni": results_uni,
      "results_bi": results_bi,
      "results_tri": results_tri
  }

  print(results)

  # with open('nlp_results.json', 'w') as json_file:
  #     json.dump(results, json_file)
  return "Hello from NLP"

@app.route("/NLPdummy", methods=["GET", "POST"])
def nlp_dummy():
  if request.method == "GET":
    return "NLP Dummy will live here."
  if request.method == "POST":
    
    # print(request.get_json())
    nlp_analysis = {
      "num_jobs_compared": 3,
      "job_id_list": ["eyJqb2JfdGl0bGUiOiJVWCBEZXNpZ25lciIsImNvbXBhbnlfbmFtZSI6IkFtYXpvbiIsImNvbXBhbnlfbWlkIjoiL20vMG1na2ciLCJhZGRyZXNzX2NpdHkiOiJOZXcgWW9yayIsImFkZHJlc3Nfc3RhdGUiOiJOZXcgWW9yayIsImh0aWRvY2lkIjoiWW1lQlJvWGo0eWYwWGgzckFBQUFBQT09IiwiaGwiOiJlbiIsImZjIjoiRXNzQkNvd0JRVTVWWDA1eFVtSlhjbnBDTkZScWMzbGxSRk13ZW5od09XOXpjekJhYTJwWlYzRXpZVEpPUldKclQyMXlWalpzVWtWdmRWbHRSRkp6UkdsNlRFRlFhMDAwT0hCWU1YbE5iVlJvUWt3NVlqSmFXalJIUTBwZk5FUXlURWRaUzNoM2FuQldNME5LYjBaSFVqZEhUMHhvYVZCT1NVeDVVM3B4WlMxVVpuUkhibWhzYnpReVZrVlRPVGswUzNrU0ZsUllOMWhaVGtoaFN6aGhUM1JSV0ZZNWNIWkpRMEVhSWtGUFRWbFNkMEU0VEdscFVGbHpaRTlQWTFCMVducExTMVJFYjA5b2VFZGpObmMiLCJmY3YiOiIzIiwiZmNfaWQiOiJmY18xNCIsImFwcGx5X2xpbmsiOnsidGl0bGUiOiJBcHBseSBvbiBMaW5rZWRJbiIsImxpbmsiOiJodHRwczovL3d3dy5saW5rZWRpbi5jb20vam9icy92aWV3L3V4LWRlc2lnbmVyLWF0LWFtYXpvbi0yNTMyMjExODgzP3V0bV9jYW1wYWlnbj1nb29nbGVfam9ic19hcHBseVx1MDAyNnV0bV9zb3VyY2U9Z29vZ2xlX2pvYnNfYXBwbHlcdTAwMjZ1dG1fbWVkaXVtPW9yZ2FuaWMifX0=",
      "eyJqb2JfdGl0bGUiOiJKdW5pb3IgVVggRGVzaWduZXIiLCJjb21wYW55X25hbWUiOiJCcmlsbGlhbnQgRXhwZXJpZW5jZSIsImNvbXBhbnlfbWlkIjoiL2cvMTFjNzN0MnY3cCIsImFkZHJlc3NfY2l0eSI6Ik5ldyBZb3JrIiwiYWRkcmVzc19zdGF0ZSI6Ik5ldyBZb3JrIiwiaHRpZG9jaWQiOiI5bkJ6aFNvSTBEdU10WjB0QUFBQUFBPT0iLCJobCI6ImVuIiwiZmMiOiJFdllCQ3JjQlFVNVZYMDV4VW05Zk5FVnVZMWRyYldwUlNYQkRiR3RrTUhjemVreFpjRlpXTTNwVVMwVTFaRmxKZDJGRmNVcEVlVGM0UzI5R1RWb3hVM05NV21nMmFEUjRSSFZVUmxVNVRqUXhiWGxFTkVKcWRVSnVlWGRaTm01NU5rSmtObmsxZDA1V1JHSXlkRGN4WlVOeWRrdDZlSFZuV2xOdmFsZE9XbXBuY0RaU01EazVhMGRaU0ZoMVVsaE1SREZtYzFOaFJIWmliVE51Vm10d2NqRkllWEUxUVdVMVRVMVpUamhHY1dwbk5WbGhMVUZaTUZKbFR6a3dFaFpVV0RkWVdVNUlZVXM0WVU5MFVWaFdPWEIyU1VOQkdpSkJUMDFaVW5kRVdrRTBPWEpyWjB0NVZYVm5iMUJvWlhWa0xVVlVMV0pYVWpGUiIsImZjdiI6IjMiLCJmY19pZCI6ImZjXzIxIiwiYXBwbHlfbGluayI6eyJ0aXRsZSI6IkFwcGx5IG9uIEdsYXNzZG9vciIsImxpbmsiOiJodHRwczovL3d3dy5nbGFzc2Rvb3IuY29tL2pvYi1saXN0aW5nL2p1bmlvci11eC1kZXNpZ25lci1icmlsbGlhbnQtZXhwZXJpZW5jZS1KVl9JQzExMzIzNDhfS08wLDE4X0tFMTksMzkuaHRtP2psPTM3MDUyOTEwNDlcdTAwMjZ1dG1fY2FtcGFpZ249Z29vZ2xlX2pvYnNfYXBwbHlcdTAwMjZ1dG1fc291cmNlPWdvb2dsZV9qb2JzX2FwcGx5XHUwMDI2dXRtX21lZGl1bT1vcmdhbmljIn19",
      "eyJqb2JfdGl0bGUiOiJVWCBEZXNpZ25lciIsImNvbXBhbnlfbmFtZSI6IlJvYmVydCBIYWxmIiwiY29tcGFueV9taWQiOiIvbS8wN2s5OG0iLCJhZGRyZXNzX2NpdHkiOiJOZXcgWW9yayIsImFkZHJlc3Nfc3RhdGUiOiJOZXcgWW9yayIsImh0aWRvY2lkIjoiNUE5ZkczMHBUNkg3UDZuNEFBQUFBQT09IiwiaGwiOiJlbiIsImZjIjoiRXVFQkNxSUJRVTVWWDA1eFZFVmpaVFJNTTIxaU1EVktkVkZ1WlVWMk1UTk9WSFZIVFhkVlpUWnBURmxUVm5wMVF6ZFdaSEYwVFRCdmRtd3hSMjlzZGxwWlIyOVlZMjFsWkhFM1ZFbFVORTB3U1dWd01EZERibUY1YVVoUVREQnZOMFZ2YlZSMmNHcENSRlJNWjNsRU1uQkNRMDR4Tm1sMFdXTXhPSEpsWVRRNVp6UnRVWGgxVEd4ek5ucERTbVExTVdFMWVIQkVjelpYWkcxbFkyMUNjVnA0TWt4V1ZYRm5FaFpVV0RkWVdVNUlZVXM0WVU5MFVWaFdPWEIyU1VOQkdpSkJUMDFaVW5kQ2FFYzViVW81YWxBMGFqTnlUM0EzTWtoT2MyNTZWWHBST1VsUiIsImZjdiI6IjMiLCJmY19pZCI6ImZjXzM2IiwiYXBwbHlfbGluayI6eyJ0aXRsZSI6IkFwcGx5IG9uIFJvYmVydCBIYWxmIiwibGluayI6Imh0dHBzOi8vd3d3LnJvYmVydGhhbGYuY29tL2pvYi9uZXcteW9yay1ueS91eC1kZXNpZ25lci8wNDgzNS0wMDExNzgwNTIxLXVzZW4\/dXRtX2NhbXBhaWduPWdvb2dsZV9qb2JzX2FwcGx5XHUwMDI2dXRtX3NvdXJjZT1nb29nbGVfam9ic19hcHBseVx1MDAyNnV0bV9tZWRpdW09b3JnYW5pYyJ9fQ=="],
      "results_uni": ['user', 'will', 'client', 'need', 'solution'],
      "results_bi": ['usercentered design', 'attention detail', 'user research', 'digital product', 'design system'],
      "results_tri": ['great attention detail', 'work within exist', 'mockups prototype effectively', 'prototype effectively communicate', 'product provide solution']
    }

    #assign variable to request.get_json
    #use this as object to pass to python script
    #return object

    #for now will spoof with sending back json and declaring nlp analysis return data hardcoded 

    return nlp_analysis
#app .run
if __name__ == "__main__":
  app.run()


