from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin


application = Flask(__name__)
cors = CORS(application, resources={r"/": {"origins": "*"}})
application.config['CORS_HEADERS'] = 'Content-Type'

@application.route('/', methods=['POST','OPTIONS'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def getText():
    res_text = request.get_json(force=True)
    new_text = {'received': res_text,
                "about":"Hey there!",
                 "group1": "Emotional issues support group",
                 "group2": "Mental issues support group",
                 "group3": "Mood issues support group",
                 "group4": "Depression issues support group",
                 "group5": "All kind of issues support group"}
    response = jsonify(new_text)
    return response

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
    application.run(debug=True)



