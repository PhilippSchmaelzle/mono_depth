from __future__ import print_function
import grpc
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import numpy as np
import cv2
import time



# Use to get information of input and output
"""
with grpc.insecure_channel("localhost:8500") as channel:
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = get_model_metadata_pb2.GetModelMetadataRequest(
      model_spec=model_pb2.ModelSpec(name="models2serv"),
      metadata_field=["signature_def"])

  response = stub.GetModelMetadata(request)
  sigdef_str = response.metadata["signature_def"].value

  print ("Name:", response.model_spec.name)
  print ("Version:", response.model_spec.version.value)
  print (get_model_metadata_pb2.SignatureDefMap.FromString(sigdef_str))
"""

# Dummy input data for batch size 3.

image_file = "/home/fascar/Documents/mono_depth_inference/test_image.png"
input_image = cv2.imread(image_file)
input_image = cv2.resize(input_image, (416,128))
batch_input = np.array(input_image, dtype='float32')
print(batch_input.shape)

channel_opt = [('grpc.max_send_message_length', 2*1024*2046*3), ('grpc.max_receive_message_length', 2*1024*2046*3)]
with grpc.insecure_channel("localhost:7500", options=channel_opt) as channel:
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest(
      model_spec=model_pb2.ModelSpec(name="models_serv"),
      inputs={"input_data": tf.make_tensor_proto(batch_input)})

  start_time =  time.time()
  NUMBER_RUNS = 100
  for i in range(NUMBER_RUNS):
    response = stub.Predict(request)
    batch_output = tf.make_ndarray(response.outputs["depth_prediction_output"])
end_time = time.time()
fps = 1 / ((end_time - start_time)/NUMBER_RUNS)

print("FPS: " + str(fps))
