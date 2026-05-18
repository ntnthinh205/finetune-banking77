import grpc
from concurrent import futures
import sys

# Import file pb2 được sinh ra TỪ FILE intent_service.proto của Backend
from protos import intent_service_pb2
from protos import intent_service_pb2_grpc
from scripts.inference import IntentClassification

MODEL_PATH = "/content/drive/MyDrive/Năm 3/Kỳ 2/Ứng dụng xử lý ngôn ngữ tự nhiên trong doanh nghiệp /Bài tập/BT02: Finetune (unsloth)/checkpoints"

# Thừa kế đúng class Servicer từ file grpc mới
class IntentServer(intent_service_pb2_grpc.IntentServiceServicer):
    def __init__(self):
        print("Loading Model...")
        self.model = IntentClassification(model_path=MODEL_PATH)
        print("Model Ready!")

    # Tên hàm phải giống hệt như trong intent_service.proto (IntentRecognizer)
    def IntentRecognizer(self, request, context):
        # Lấy request.message (chứ không phải request.text)
        pred = self.model(request.message)
        print(f"Message: {request.message} -> Intent: {pred}")
        
        # Trả về kết quả, thêm confidence (theo yêu cầu đề bài)
        # Nếu model của bạn có xuất ra confidence thì thay vào 0.99, nếu không thì hardcode tạm
        return intent_service_pb2.IntentResponse(
            intent=pred, 
            confidence=0.99, 
            reason="Predicted by unsloth model"
        )

def start_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    # Đăng ký class Servicer vào server
    intent_service_pb2_grpc.add_IntentServiceServicer_to_server(IntentServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Server listening on 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    start_server()
