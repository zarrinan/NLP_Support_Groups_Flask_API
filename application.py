import pandas as pd
import numpy as np
import pandas as pd
import json
from flask import Flask, abort, jsonify, request
import _pickle as pickle
from sklearn import utils
from sklearn.linear_model import LogisticRegression
import gensim
import gensim.models.doc2vec as doc2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re

from flask import Flask, abort, jsonify, request

from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin

# pickled multiple regression model
#pipe = load('pipeline.joblib')
# pickled multiple regression model
clf = pickle.load(open('clf.pkl','rb'))
model_dbow = Doc2Vec.load('doc2vec_model_reddit.pkl')

application = app = Flask(__name__)   # AWS EBS expects variable name "application"

cors = CORS(application, resources={r"/": {"origins": "*"}})
application.config['CORS_HEADERS'] = 'Content-Type'

@application.route('/', methods=['POST','OPTIONS'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])

def make_predict():
    # read in data
    user_text = request.get_json(force=True)

    # get variables
    text_input = user_text['text']


    # combine
    text = re.sub(r'http\S+|www.\S+', 'link', text_input)
    text = text.split(' ')
    #text = list(text)

    # label sentences
    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
        return labeled

    # function to vectorize input data
    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    # prep text for prediction
    user_input = model_dbow.infer_vector(text, steps=20).reshape(1, -1)

    # make prediction and convert it to list so that jsonify is happy
    output = pd.DataFrame(clf.predict_proba(user_input), columns=clf.classes_).T.nlargest(5, [0])[0].reset_index().values.tolist()

    # send back the top 5 subreddits and their associated probabilities
    return jsonify(support_groups = output)

@application.route('/')
def findGroups():
  return jsonify(
    {"about":"Hey there!",
     "group1": "Emotional issues support group",
     "group2": "Mental issues support group",
     "group3": "Mood issues support group",
     "group4": "Depression issues support group",
     "group5": "All kind of issues support group"
    })

if __name__ == '__main__':
    application.run(port = 8080, debug = True)


