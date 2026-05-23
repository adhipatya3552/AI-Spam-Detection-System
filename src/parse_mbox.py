import os
import sys
import mailbox
import re
import pandas as pd

# Add the project root and src/ to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "src"))

from predict import extract_email_text

def anonymize_text(text, user_names=["adhipatya", "saxena", "adhip"]):
    # Replace email addresses
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[email]', text)
    
    # Replace URLs
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    text = re.sub(url_pattern, '[url]', text)
    
    # Replace phone numbers
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    text = re.sub(phone_pattern, '[phone]', text)
    
    # Replace names
    for name in user_names:
        text = re.compile(re.escape(name), re.IGNORECASE).sub('[name]', text)
        
    return text

def main():
    mbox_path = os.path.join(base_dir, "data", "Spam.mbox")
    output_csv = os.path.join(base_dir, "data", "anonymized_spam.csv")
    
    print(f"Reading mailbox from {mbox_path}...")
    if not os.path.exists(mbox_path):
        print(f"Error: {mbox_path} does not exist!")
        sys.exit(1)
        
    mbox = mailbox.mbox(mbox_path)
    total_messages = len(mbox)
    print(f"Found {total_messages} messages in mailbox.")
    
    processed_emails = []
    
    for idx, msg in enumerate(mbox):
        try:
            # Convert to raw message string
            msg_str = msg.as_string()
            
            # Extract plain text content
            cleaned = extract_email_text(msg_str)
            
            # Anonymize personal info
            anonymized = anonymize_text(cleaned)
            
            # Append to list with spam label (1)
            processed_emails.append({
                "label": "spam",
                "text": anonymized
            })
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_messages:
                print(f"Processed {idx + 1}/{total_messages} messages...")
        except Exception as e:
            print(f"Warning: Failed to parse message {idx}: {e}")
            
    # Write to CSV
    df = pd.DataFrame(processed_emails)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Successfully saved {len(df)} anonymized spam messages to {output_csv}.")

if __name__ == "__main__":
    main()
