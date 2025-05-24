import spacy
import re
import dateparser
from nlp.summarizer import generate_summary

nlp = spacy.load("en_core_web_sm")

INTENT_CATEGORIES = {
    "reminder": ["remind", "set reminder", "alarm"],
    "summary": ["summarize", "recap", "brief"],
    "project_allocation": ["assign", "allocate", "project"]
}

def classify_intent(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc]

    for intent, keywords in INTENT_CATEGORIES.items():
        for keyword in keywords:
            if all(word in lemmas for word in keyword.split()):
                return intent

    for token in doc:
        if token.pos_ == "VERB":
            return token.lemma_

    return "unknown"


def extract_reminder_info(text):
    parsed_time = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
    time = parsed_time.strftime("%I:%M %p") if parsed_time else None

    if not time:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "TIME":
                time = ent.text
                break

    if not time:
        time_match = re.search(r"\b(?:at\s*)?(\d{1,2}(:\d{2})?\s*(am|pm|AM|PM)?)\b", text, re.IGNORECASE)
        if time_match:
            time = time_match.group(1).upper()

    doc = nlp(text)
    subject_tokens = [t.text for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    subject = " ".join(subject_tokens).strip() if subject_tokens else None

    if not subject:
        subject_chunks = list(doc.noun_chunks)
        if subject_chunks:
            subject = subject_chunks[-1].text

    return {
        "intent": "reminder",
        "time": time,
        "topic": subject
    }


def extract_project_allocation_info(text):
    doc = nlp(text)

    project_name = None
    project_lead = None
    description = None

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not project_lead:
            project_lead = ent.text
        elif ent.label_ in {"ORG", "PRODUCT"} and not project_name:
            project_name = ent.text

    if not project_name:
        capital_nouns = [t.text for t in doc if t.pos_ == "PROPN"]
        if capital_nouns:
            project_name = capital_nouns[0]

    desc_tokens = [t.text for t in doc if t.pos_ in {"ADJ", "NOUN"} and t.text != project_name]
    if desc_tokens:
        description = " ".join(desc_tokens)

    return {
        "intent": "project_allocation",
        "project_name": project_name,
        "project_lead": project_lead,
        "description": description
    }


def extract_summary_info(text):
    return {
        "intent": "summary",
        "summary": generate_summary(text)
    }

def process_input(text):
    intent = classify_intent(text)

    if intent == "reminder":
        return extract_reminder_info(text)
    elif intent == "project_allocation":
        return extract_project_allocation_info(text)
    elif intent == "summary":
        return extract_summary_info(text)
    else:
        return {"intent": intent, "message": "Dynamic intent, no extractor yet"}
