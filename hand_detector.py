import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, frame, draw=True):
        """Detect hands in the frame and optionally draw landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return frame
    
    def get_landmark_positions(self, frame):
        """Get normalized positions of hand landmarks."""
        landmark_positions = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]  # Get first hand
            for landmark in hand.landmark:
                landmark_positions.append((landmark.x, landmark.y))
        return landmark_positions
    
    def hands_detected(self):
        """Return True if any hands are detected."""
        return bool(self.results.multi_hand_landmarks)