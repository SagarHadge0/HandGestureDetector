import cv2
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer
from word_builder import WordBuilder

def main():
    # Initialize components
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer()
    word_builder = WordBuilder()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Detect hands and get landmarks
        frame = hand_detector.find_hands(frame)
        landmarks = hand_detector.get_landmark_positions(frame)
        
        if landmarks:
            # Recognize gesture
            gesture = gesture_recognizer.predict(frame)
            if gesture:
                word_builder.add_letter(gesture)
                # Display the recognized gesture
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
        # Display current word
        current_word = word_builder.get_word()
        cv2.putText(
            frame,
            f"Word: {current_word}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Show frame
        cv2.imshow("Hand Gesture Recognition", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            word_builder.clear_word()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()