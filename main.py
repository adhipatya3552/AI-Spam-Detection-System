from src.train_model import train
from src.evaluate_model import evaluate
from src.predict import predict_message

def main():

    while True:

        print("\n===== AI SPAM DETECTION SYSTEM =====\n")

        print("1. Train Model")
        print("2. Evaluate Model")
        print("3. Predict Message")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        # Train
        if choice == "1":

            train()

        # Evaluate
        elif choice == "2":

            evaluate()

        # Predict
        elif choice == "3":

            message = input("\nEnter your message:\n")

            predict_message(message)

        # Exit
        elif choice == "4":

            print("\nExiting project...")

            break

        else:

            print("\nInvalid choice. Try again.")


if __name__ == "__main__":

    main()