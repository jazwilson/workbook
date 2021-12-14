FROM tensorflow/serving:2.7.0

COPY view-model /models/vaccination-model/1
ENV MODEL_NAME='vaccination-model'
