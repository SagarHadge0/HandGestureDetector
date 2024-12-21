import os

def check_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "gesture_recognizer.task"))
    
    print(f"Current directory: {current_dir}")
    print(f"Model path: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")

if __name__ == "__main__":
    check_model() 