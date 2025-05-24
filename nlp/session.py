import spacy

nlp_model = spacy.load("en_core_web_sm")  # Load once globally

class nlp_userInput:
    def __init__(self):
        self._data = []

    def set(self, value: str):
        # 1. Simple check: if the exact string already exists, do not add it
        if value in self._data:
            return

        # 2. Semantic similarity check with the last entry
        if self._data:
            last_doc = nlp_model(self._data[-1])
            new_doc = nlp_model(value)

            similarity = last_doc.similarity(new_doc)

            if similarity > 0.85:  # Threshold can be tuned
                self._data[-1] = value  # Replace the last input with the more detailed one
                return

        # If it's new and dissimilar, append normally
        self._data.append(value)

    def get(self):
        return self._data

    def clear(self):
        self._data = []
        print("Session has been cleared.")
