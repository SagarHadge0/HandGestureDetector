import os

def check_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "gesture_recognizer.task")
    model_path = os.path.normpath(model_path)
    
    print(f"Current directory: {current_dir}")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        print(f"Model size: {os.path.getsize(model_path)} bytes")

if __name__ == "__main__":
    check_model() 