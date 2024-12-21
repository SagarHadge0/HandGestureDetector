import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import time

class GestureRecognizer:
    def __init__(self):
        try:
            # Get absolute path to the model file
            model_path = os.path.abspath("gesture_recognizer.task")
            
            print(f"Loading model from: {model_path}")
            
            # Create base options with the correct path
            base_options = python.BaseOptions(model_asset_buffer=open(model_path, 'rb').read())
            
            # Create options for the recognizer with optimized parameters
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_hand_detection_confidence=0.2,  # Lower threshold for better detection
                min_hand_presence_confidence=0.2,
                min_tracking_confidence=0.2,
                num_hands=1
            )
            
            # Create the recognizer
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print("Gesture recognizer initialized successfully!")
            
            # Store last gesture and confidence
            self.last_gesture = None
            self.last_confidence = 0
            self.gesture_counter = 0
            self.GESTURE_THRESHOLD = 2  # Reduced threshold for faster response
            self.MIN_CONFIDENCE = 0.2  # Minimum confidence threshold
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def predict(self, frame):
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect gestures
            recognition_result = self.recognizer.recognize(mp_image)
            
            if recognition_result.gestures and recognition_result.handedness:
                # Get the top gesture and its confidence
                gesture = recognition_result.gestures[0][0].category_name
                confidence = recognition_result.gestures[0][0].score
                handedness = recognition_result.handedness[0][0].category_name
                
                # Print all detected gestures with confidence
                print("\nDetected gestures:")
                for gesture_list in recognition_result.gestures:
                    for g in gesture_list:
                        print(f"- {g.category_name}: {g.score:.2f}")
                
                # Only update if confidence is above threshold
                if confidence > self.MIN_CONFIDENCE:
                    if gesture == self.last_gesture:
                        self.gesture_counter += 1
                    else:
                        self.gesture_counter = 0
                        
                    # Update gesture if we've seen it consistently
                    if self.gesture_counter >= self.GESTURE_THRESHOLD:
                        self.last_gesture = gesture
                        self.last_confidence = confidence
                        print(f"\nConfirmed gesture: {gesture}")
                        print(f"Hand: {handedness}")
                        print(f"Confidence: {confidence:.2f}")
                        return gesture
                        
                    self.last_gesture = gesture
                    return self.last_gesture if self.last_gesture else None
                
            return self.last_gesture if self.last_gesture else None
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None