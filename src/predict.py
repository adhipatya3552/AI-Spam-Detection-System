import os
import sys
import email
from email import policy
import re
import joblib
from html.parser import HTMLParser

# Determine base directory dynamically to ensure absolute paths work
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure we can import modules in src/ relative to base_dir
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
if os.path.join(base_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(base_dir, "src"))

from preprocess import clean_text

# Load trained model using absolute path
model_path = os.path.join(base_dir, "models", "spam_pipeline.pkl")
model = joblib.load(model_path)


class EmailHTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.hidden_fed = []
        self.hide_stack = []
        self.ignored_tags = {'style', 'script', 'object', 'embed', 'template', 'head'}

    def handle_starttag(self, tag, attrs):
        is_ignored = tag.lower() in self.ignored_tags
        is_hidden = False
        attrs_dict = dict(attrs)
        style_attr = attrs_dict.get('style', '').replace(' ', '').lower()
        if 'display:none' in style_attr:
            is_hidden = True
        if is_ignored or is_hidden or self.hide_stack:
            self.hide_stack.append(tag)

    def handle_endtag(self, tag):
        if self.hide_stack:
            if tag in self.hide_stack:
                while self.hide_stack and self.hide_stack[-1] != tag:
                    self.hide_stack.pop()
                if self.hide_stack:
                    self.hide_stack.pop()

    def handle_data(self, data):
        if not self.hide_stack:
            self.fed.append(data)
        else:
            if self.hide_stack and self.hide_stack[-1] not in {'style', 'script', 'head'}:
                self.hidden_fed.append(data)

    def get_data(self):
        return ''.join(self.fed)

    def get_hidden_data(self):
        return ''.join(self.hidden_fed)


def parse_email_message(raw_content):
    """
    Parses dynamic raw email content.
    Returns (cleaned_text, hidden_text_len, subject, body).
    """
    hidden_len = 0
    subject = ""
    body = ""
    from_val = ""
    
    try:
        msg = email.message_from_string(raw_content, policy=policy.default)
        if not msg.keys():
            return raw_content, 0, "", raw_content
            
        subject = msg.get('subject', '') or ''
        from_val = msg.get('from', '') or ''
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get_content_disposition())
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                    break
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                
        if not body.strip():
            html_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/html':
                        html_body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                        break
            else:
                if msg.get_content_type() == 'text/html':
                    html_body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                    
            if html_body.strip():
                stripper = EmailHTMLStripper()
                stripper.feed(html_body)
                body = stripper.get_data()
                body = re.sub(r'\s+', ' ', body).strip()
                hidden_len = len(stripper.get_hidden_data().strip())
                
        if subject or body:
            components = []
            if from_val:
                components.append(f"From: {from_val}")
            if subject:
                components.append(f"Subject: {subject}")
            if body:
                components.append(body.strip())
            return "\n\n".join(components), hidden_len, subject, body
            
    except Exception:
        pass
        
    return raw_content, 0, "", raw_content


def check_heuristics(subject, body, hidden_text_len):
    """
    Checks high-confidence heuristics for email spam/phishing.
    Returns (is_spam_heuristic, confidence_heuristic).
    """
    subject_lower = subject.lower()
    body_lower = body.lower()
    
    phishing_phrases = [
        "blocked your account",
        "renew your subscription",
        "photos and videos will be deleted",
        "account will be deleted",
        "storage has reached",
        "sync has been paused",
        "unable to renew",
        "permanently removed",
        "final notice",
        "upgrade storage",
        "unauthorized access",
        "verify your identity",
        "verify your account",
        "billing information",
        "critical limit"
    ]
    
    matches = []
    for phrase in phishing_phrases:
        if phrase in subject_lower or phrase in body_lower:
            matches.append(phrase)
            
    # Check 1: Bayesian Poisoning (Large amount of hidden text inside style=display:none or object)
    if hidden_text_len > 150:
        return True, 99.5
        
    # Check 2: Strong phishing matches in subject or body
    if len(matches) >= 2:
        return True, 97.5
        
    # Check 3: Urgent subject lines
    if any(p in subject_lower for p in ["blocked your account", "will be deleted", "renew your subscription", "final notice", "photos and videos"]):
        return True, 95.0

    return False, 0.0


def extract_email_text(raw_content):
    """
    Extracts plain text body and subject from raw MIME/SMTP email content.
    If the content does not contain typical email headers, it returns the raw content.
    """
    cleaned_text, _, _, _ = parse_email_message(raw_content)
    return cleaned_text


def predict_message(message):
    # Parse the message
    cleaned_text, hidden_text_len, subject, body = parse_email_message(message)

    # Heuristic override
    is_heur, conf_heur = check_heuristics(subject, body, hidden_text_len)
    if is_heur:
        return 1, conf_heur

    # Clean message using existing clean_text logic
    cleaned_message = clean_text(cleaned_text)

    # Prediction
    prediction = model.predict([cleaned_message])[0]

    # Probability
    probability = model.predict_proba([cleaned_message])[0]

    # Confidence
    if prediction == 1:
        confidence = probability[1] * 100
    else:
        confidence = probability[0] * 100

    return prediction, confidence


if __name__ == "__main__":

    user_message = input("Enter your message:\n")

    prediction, confidence = predict_message(user_message)

    if prediction == 1:

        print("\n🚨 SPAM MESSAGE DETECTED")

    else:

        print("\n✅ SAFE MESSAGE")

    print(f"Confidence: {confidence:.2f}%")