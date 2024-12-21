import cv2
import numpy as np
import json
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer

class TrainingDataCollector:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.samples = []
        self.labels = []
        
    def collect_samples(self, letter, num_samples=100):
        """Collect training samples for a specific letter."""
        cap = cv2.VideoCapture(0)
        collected = 0
        
        while collected < num_samples:
            success, frame = cap.read()
            if not success:
                break
                
            frame = self.hand_detector.find_hands(frame)
            landmarks = self.hand_detector.get_landmark_positions(frame)
            
            if landmarks:
                self.samples.append(landmarks)
                self.labels.append(letter)
                collected += 1
            
            # Display progress
            cv2.putText(
                frame,
                f"Collecting {letter}: {collected}/{num_samples}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def save_data(self, filename):
        """Save collected training data to a file."""
        data = {
            'samples': [[(x, y) for x, y in sample] for sample in self.samples],
            'labels': self.labels
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
            
    def load_data(self, filename):
        """Load training data from a file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.samples = [[(x, y) for x, y in sample] for sample in data['samples']]
        self.labels = data['labels']

def main():
    collector = TrainingDataCollector()
    
    # Collect samples for thumb gesture
    print("Prepare to show thumb gesture...")
    print("Press any key to start collecting...")
    cv2.waitKey(0)
    collector.collect_samples("thumb", num_samples=50)  # You can adjust number of samples
    
    # Save the collected data
    collector.save_data("gesture_training_data.json")
    print("Training data saved!")

if __name__ == "__main__":
    main()