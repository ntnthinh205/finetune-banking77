import grpc
from concurrent import futures
import sys
import threading
import time

# Update these paths based on where you cloned the repo in Colab
PROJECT_PATH = '/content/intent_model' # Adjust this to your repo name
sys.path.append(f"{PROJECT_PATH}/scripts")

import intent_pb2
import intent_pb2_grpc
from inference import IntentClassification

class IntentServer(intent_pb2_grpc.IntentServiceServicer):
    def __init__(self):
        print("Loading Model...")
        self.model = IntentClassification(model_path=f"{PROJECT_PATH}/checkpoints")
        print("Model Ready!")

    def PredictIntent(self, request, context):
        pred = self.model(request.text)
        print(f"Text: {request.text} -> Intent: {pred}")
        return intent_pb2.IntentResponse(intent=pred)

def start_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    intent_pb2_grpc.add_IntentServiceServicer_to_server(IntentServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Server listening on 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    start_server()
