import flask
from flask import Flask, request,jsonify
import pickle
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = Flask(__name__)



@app.route('/predict',methods=['GET','POST'])
def predict():
  tfidf = pickle.load(open('tfidf', 'rb'))
  reques = request.form['headline']
  result = search(tfidf,vector, reques, top_n = 5)
  res=print_result(reques,result,X)
  return(jsonify(res))

def print_result(request_content,result,X):
    print('\nsearch : ' + request_content)
    print('\nBest Results :')
    for i in result:
        #s={'id = {0:5d} - headline = {1}'.format(i, X['headline'].loc[i]), 'link={1}'.format(i,X['link'].loc[i])}
        out=[]
        str={'id': i, 'headline': X['headline'].loc[i],'link': X['link'].loc[i]}
        out.append(str)
        print(out)

def search(tfidf_matrix,model,reques, top_n = 5):
    request_transform = model.transform([reques])
    similarity = np.dot(request_transform,np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[-5:][::-1]
    return indices

if __name__ == '__main__':
    X = pickle.load(open('X', 'rb'))
    vector = pickle.load(open('vector', 'rb'))
    tfidf = pickle.load(open('tfidf', 'rb'))
    app.run(port=9000, debug=True)