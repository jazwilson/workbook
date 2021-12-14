import tensorflow as tf

import pickle
import os

from flask      import Flask, request, jsonify
from protobuf_to_dict import protobuf_to_dict

import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from proto import np_to_protobuf

columns = ['state_Arizona',
 'state_Arkansas',
 'state_California',
 'state_Colorado',
 'state_Connecticut',
 'state_Florida',
 'state_Idaho',
 'state_Illinois',
 'state_Iowa',
 'state_Maine',
 'state_Massachusetts',
 'state_Michigan',
 'state_Minnesota',
 'state_Missouri',
 'state_Montana',
 'state_New Jersey',
 'state_New York',
 'state_North Carolina',
 'state_North Dakota',
 'state_Ohio',
 'state_Oklahoma',
 'state_Oregon',
 'state_Pennsylvania',
 'state_Rhode Island',
 'state_South Dakota',
 'state_Tennessee',
 'state_Texas',
 'state_Utah',
 'state_Vermont',
 'state_Virginia',
 'state_Washington',
 'state_Wisconsin',
 'type_0',
 'type_BOCES',
 'type_Charter',
 'type_Kindergarten',
 'type_Nonpublic',
 'type_Private',
 'type_Public']

dv  = pickle.load(open('model.dv','rb'))


host = os.getenv('TF_SERVING_HOST','localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

app = Flask('vaccination')

def get(query):
    query = { k:query.get(k,0) for k in columns }
    X=dv.fit_transform([query])

    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'vaccination-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['normalization_input'].CopyFrom(np_to_protobuf(X))

    pb_response = stub.Predict(pb_request,timeout=20.0)
    preds = pb_response.outputs['dropout'].float_val
    
    print(preds)
    result = {
        'predicted_vaccination': list(preds)[0]
    }
    return result

@app.route('/predict', methods=['POST'])
def predict():
    query = request.get_json()
    result = get(query)
    return jsonify(result)

if __name__ == "__main__":
    #school = {
    # 'type_Public': 1,
    # 'type_Kindergarten':1,
    # 'state_Texas':1
    #}
    #get(school)
    app.run(debug=True, host='0.0.0.0', port=9696)