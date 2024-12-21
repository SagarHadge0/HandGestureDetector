import requests
import os

def download_model():
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "gesture_recognizer.task"))
    
    print(f"Downloading model to: {model_path}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        print(f"Model downloaded successfully to: {model_path}")
        print(f"File size: {os.path.getsize(model_path)} bytes")
        
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model() 