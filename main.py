from nlp.session import nlp_userInput
from nlp.nlp_engine import process_input, append_validation_message  # Assuming your NLP logic is in this file

def main():
    session = nlp_userInput()

    while True:
        user_input = input("Enter a string (or type 'get' to view session, 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'get':
            print("Session data:", session.get())
        else:
            # Step 1: Add to session
            session.set(user_input)

            # Step 2: Get full session context as string
            full_text = " ".join(session.get())

            # Step 3: Process through NLP
            result = process_input(full_text)

            # Step 4: Print structured result
            print("Extracted Info:", result)

if __name__ == "__main__":
    main()
