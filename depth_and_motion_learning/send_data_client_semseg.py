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
with grpc.insecure_channel("localhost:8500") as channel:
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = get_model_metadata_pb2.GetModelMetadataRequest(model_spec=model_pb2.ModelSpec(name="models2serv"), metadata_field=["signature_def"])
    response = stub.GetModelMetadata(request)
    sigdef_str = response.metadata["signature_def"].value
    print("Name:", response.model_spec.name)
    print("Version:", response.model_spec.version.value)
    print(get_model_metadata_pb2.SignatureDefMap.FromString(sigdef_str))

# Dummy input data for batch size 3.

image_file = "/home/fascar/Documents/mono_depth_inference/test_image.png"
input_image = cv2.imread(image_file)
input_image = cv2.resize(input_image, (2048, 1024))
batch_input = np.array(input_image, dtype='float32')
print(batch_input.shape)
batch_input = np.array([batch_input[:,:,i] for i in range(3)])
print(batch_input.shape)
batch_input = np.expand_dims(batch_input, axis=0)
print(batch_input.shape)

channel_opt = [('grpc.max_send_message_length', 200*1024*2048*3), ('grpc.max_receive_message_length', 200*1024*2048*3)]
with grpc.insecure_channel("localhost:8500", options=channel_opt) as channel:
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest(
    model_spec=model_pb2.ModelSpec(name="models2serv"),
    inputs={"input.1": tf.make_tensor_proto(batch_input)})
    start_time =  time.time()
    NUMBER_RUNS = 1
    for i in range(NUMBER_RUNS):
        response = stub.Predict(request)
        output_0 = np.squeeze(tf.make_ndarray(response.outputs["output_0"]), axis=0)
        output_2 = np.squeeze(tf.make_ndarray(response.outputs["output_2"]), axis=0)
    end_time = time.time()
    fps = 1 / ((end_time - start_time)/NUMBER_RUNS)

print(output_0.shape)
for i in range(19):
    gray_img = np.array(output_0[i,:,:], dtype="uint8")
    cv2.imshow("gray", gray_img)
    cv2.waitKey(0)

for i in range(256):
    gray_img = np.array(output_2[i,:,:], dtype="uint8")
    cv2.imshow("gray", gray_img)
    cv2.waitKey(10)





print("FPS: " + str(fps))