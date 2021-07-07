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
    print(request.get_json())
    return "Request JSON"
#app .run
if __name__ == "__main__":
  app.run()


