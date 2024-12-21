class WordBuilder:
    def __init__(self):
        self.current_word = []
        self.last_letter = None
        self.letter_confidence = 0
        self.min_confidence = 3  # Number of consistent predictions needed
        
    def add_letter(self, letter):
        """Add a letter to the current word with debouncing."""
        if letter == self.last_letter:
            self.letter_confidence += 1
        else:
            self.last_letter = letter
            self.letter_confidence = 1
            
        if self.letter_confidence == self.min_confidence:
            self.current_word.append(letter)
            self.letter_confidence = 0
            
    def get_word(self):
        """Get the current word."""
        return ''.join(self.current_word)
        
    def clear_word(self):
        """Clear the current word."""
        self.current_word = []
        self.last_letter = None
        self.letter_confidence = 0