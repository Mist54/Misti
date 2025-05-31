import spacy
import re
import dateparser
from datetime import datetime
from nlp.summarizer import generate_summary  # Assume your custom summarizer

# Load spaCy model globally for efficiency
nlp = spacy.load("en_core_web_sm")

# Comprehensive intent classification keywords
INTENT_CATEGORIES = {
    "reminder": {
        "keywords": ["remind", "reminder", "set reminder", "alarm", "notify", "notification",
                     "alert", "schedule reminder", "wake me", "don't forget"],
        "context_words": ["time", "date", "when", "at", "on", "tomorrow", "today", "later"]
    },
    "summary": {
        "keywords": ["summarize", "summary", "recap", "brief", "overview", "digest",
                     "abstract", "condensed", "key points", "main points", "gist"],
        "context_words": ["document", "text", "article", "report", "content", "information"]
    },
    "project_allocation": {
        "keywords": ["assign", "allocate", "project", "delegate", "task", "assignment",
                     "responsibility", "project allocation", "team assignment", "work allocation"],
        "context_words": ["team", "member", "lead", "manager", "person", "developer", "designer"]
    },
    "meeting": {
        "keywords": ["meeting", "meet", "conference", "call", "discussion", "session",
                     "standup", "sync", "review", "presentation"],
        "context_words": ["room", "time", "attendees", "agenda", "zoom", "teams"]
    },
    "task": {
        "keywords": ["task", "todo", "to-do", "job", "work", "activity", "action item",
                     "deliverable", "milestone"],
        "context_words": ["complete", "finish", "done", "priority", "urgent", "deadline"]
    }
}

# Enhanced DB Action keywords with comprehensive synonyms
DB_ACTION_SYNONYMS = {
    "create": {
        "keywords": ["create", "add", "set", "schedule", "make", "new", "register", "insert",
                     "establish", "generate", "build", "setup", "start", "initiate", "begin"],
        "patterns": [r"\bcreate\b", r"\badd\b", r"\bset\s+(?:up|a)\b", r"\bmake\s+(?:a|new)\b",
                     r"\bschedule\b", r"\bnew\b"]
    },
    "read": {
        "keywords": ["read", "get", "retrieve", "show", "list", "view", "display", "find",
                     "search", "fetch", "check", "see", "look", "examine", "review"],
        "patterns": [r"\bshow\s+me\b", r"\bget\s+(?:me|the)\b", r"\blist\s+(?:all|the)\b",
                     r"\bfind\b", r"\bsearch\b", r"\bview\b", r"\bdisplay\b"]
    },
    "update": {
        "keywords": ["update", "edit", "change", "modify", "alter", "revise", "adjust",
                     "correct", "fix", "amend", "improve", "enhance", "refine", "tweak"],
        "patterns": [r"\bupdate\b", r"\bedit\b", r"\bchange\b", r"\bmodify\b",
                     r"\balter\b", r"\brevise\b", r"\badjust\b"]
    },
    "delete": {
        "keywords": ["delete", "remove", "cancel", "clear", "drop", "eliminate", "erase",
                     "destroy", "purge", "wipe", "unset", "dismiss", "discard", "abandon"],
        "patterns": [r"\bdelete\b", r"\bremove\b", r"\bcancel\b", r"\bclear\b",
                     r"\beliminate\b", r"\berase\b", r"\bdrop\b"]
    }
}

# Context patterns for better intent detection
CONTEXT_PATTERNS = {
    "question_indicators": ["what", "how", "when", "where", "who", "which", "why", "?"],
    "command_indicators": ["please", "can you", "could you", "would you", "i want", "i need"],
    "time_indicators": ["at", "on", "by", "until", "before", "after", "during", "tomorrow",
                        "today", "tonight", "morning", "afternoon", "evening", "night"],
    "ownership_indicators": ["my", "our", "his", "her", "their", "the", "this", "that"]
}


def preprocess_text(text):
    """
    Advanced text preprocessing with normalization and cleaning
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())

    # Convert common contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not", "'ll": " will",
        "'re": " are", "'ve": " have", "'d": " would", "'m": " am",
        "let's": "let us", "that's": "that is", "there's": "there is",
        "here's": "here is", "what's": "what is", "who's": "who is"
    }

    for contraction, expansion in contractions.items():
        text = re.sub(rf"\b\w*{re.escape(contraction)}\b",
                      lambda m: m.group().replace(contraction, expansion),
                      text, flags=re.IGNORECASE)

    # Normalize time expressions
    time_normalizations = {
        r"\b(\d{1,2})\s*(?:o'?clock|oclock)\b": r"\1:00",
        r"\b(\d{1,2})\.(\d{2})\b": r"\1:\2",
        r"\b(\d{1,2})\s*(am|pm)\b": r"\1:00 \2",  # FIXED: group (am|pm) captured
    }

    for pattern, replacement in time_normalizations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def detect_db_intent_first_pass(text):
    """
    First spaCy pass: Advanced DB action intent detection
    """
    if not text:
        return "create"

    doc = nlp(text.lower())
    text_lower = text.lower()

    # Score-based approach for better accuracy
    action_scores = {"create": 0, "read": 0, "update": 0, "delete": 0}

    # 1. Check explicit keywords and patterns
    for db_action, action_data in DB_ACTION_SYNONYMS.items():
        # Check keywords
        for keyword in action_data["keywords"]:
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                action_scores[db_action] += 3

        # Check patterns
        for pattern in action_data["patterns"]:
            if re.search(pattern, text_lower):
                action_scores[db_action] += 2

    # 2. Analyze grammatical structure
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            lemma = token.lemma_

            # Map verb lemmas to actions
            verb_mappings = {
                "create": ["create", "make", "add", "set", "establish", "build", "generate"],
                "read": ["show", "get", "find", "see", "look", "check", "view", "display"],
                "update": ["change", "modify", "update", "edit", "alter", "adjust", "fix"],
                "delete": ["delete", "remove", "cancel", "clear", "drop", "eliminate"]
            }

            for action, verbs in verb_mappings.items():
                if lemma in verbs:
                    action_scores[action] += 2

    # 3. Context-based inference
    if any(indicator in text_lower for indicator in CONTEXT_PATTERNS["question_indicators"]):
        action_scores["read"] += 2

    if "new" in text_lower or "setup" in text_lower:
        action_scores["create"] += 2

    if any(word in text_lower for word in ["existing", "current", "old"]):
        action_scores["update"] += 1

    # 4. Dependency analysis for complex sentences
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token.lemma_
            if root_verb in ["want", "need", "like"]:
                # Look at the object of the verb
                for child in token.children:
                    if child.dep_ in ["dobj", "xcomp"]:
                        child_lemma = child.lemma_
                        if child_lemma in ["create", "make", "add"]:
                            action_scores["create"] += 2
                        elif child_lemma in ["see", "know", "find"]:
                            action_scores["read"] += 2

    # Return the highest scoring action
    max_score = max(action_scores.values())
    if max_score > 0:
        return max(action_scores, key=action_scores.get)

    return "create"  # Default fallback


def classify_module_intent_second_pass(text, db_intent):
    """
    Second spaCy pass: Advanced module intent classification with DB context filtering
    """
    if not text:
        return "unknown"

    doc = nlp(text.lower())

    # 1. Filter out DB action words to focus on module intent
    db_action_words = set()
    for action_data in DB_ACTION_SYNONYMS.values():
        db_action_words.update(action_data["keywords"])

    # Create filtered tokens
    filtered_tokens = []
    for token in doc:
        if token.text.lower() not in db_action_words:
            filtered_tokens.append(token)

    # 2. Score-based module intent detection
    intent_scores = {}
    for intent_name in INTENT_CATEGORIES:
        intent_scores[intent_name] = 0

    # Check keywords in filtered text
    filtered_text = " ".join([token.text for token in filtered_tokens])
    filtered_doc = nlp(filtered_text.lower())

    for intent_name, intent_data in INTENT_CATEGORIES.items():
        # Check primary keywords
        for keyword in intent_data["keywords"]:
            if keyword in filtered_text.lower():
                intent_scores[intent_name] += 3

        # Check context words
        for context_word in intent_data["context_words"]:
            if context_word in filtered_text.lower():
                intent_scores[intent_name] += 1

    # 3. Named Entity Recognition for context
    entities = [(ent.text, ent.label_) for ent in filtered_doc.ents]

    for ent_text, ent_label in entities:
        if ent_label == "PERSON":
            intent_scores["project_allocation"] += 2
            intent_scores["meeting"] += 1
        elif ent_label == "TIME":
            intent_scores["reminder"] += 2
            intent_scores["meeting"] += 1
        elif ent_label == "ORG":
            intent_scores["project_allocation"] += 1

    # 4. Linguistic pattern analysis
    noun_chunks = [chunk.text.lower() for chunk in filtered_doc.noun_chunks]

    for chunk in noun_chunks:
        if any(word in chunk for word in ["meeting", "call", "conference"]):
            intent_scores["meeting"] += 2
        elif any(word in chunk for word in ["project", "task", "assignment"]):
            intent_scores["project_allocation"] += 2
        elif any(word in chunk for word in ["reminder", "alarm", "notification"]):
            intent_scores["reminder"] += 2
        elif any(word in chunk for word in ["summary", "report", "document"]):
            intent_scores["summary"] += 2

    # 5. DB intent context influence
    if db_intent == "create":
        intent_scores["reminder"] += 1
        intent_scores["project_allocation"] += 1
    elif db_intent == "read":
        intent_scores["summary"] += 1

    # Return highest scoring intent
    max_score = max(intent_scores.values()) if intent_scores else 0
    if max_score > 0:
        return max(intent_scores, key=intent_scores.get)

    return "unknown"


def extract_reminder_info(text):
    """
    Advanced reminder information extraction with multiple fallback strategies
    """
    result = {"time": None, "topic": None, "date": None, "priority": None}

    # 1. Advanced time extraction using dateparser
    try:
        parsed_time = dateparser.parse(text, settings={
            'PREFER_DATES_FROM': 'future',
            'RETURN_AS_TIMEZONE_AWARE': False,
            'DATE_ORDER': 'MDY'
        })
        if parsed_time:
            result["time"] = parsed_time.strftime("%I:%M %p")
            result["date"] = parsed_time.strftime("%Y-%m-%d")
    except Exception:
        pass

    # 2. Fallback time extraction using spaCy NER
    if not result["time"]:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["TIME", "DATE"]:
                result["time"] = ent.text
                break

    # 3. Regex-based time extraction
    if not result["time"]:
        time_patterns = [
            r"\b(?:at\s*)?(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))\b",
            r"\b(?:at\s*)?(\d{1,2}\s*(?:am|pm|AM|PM))\b",
            r"\b(tomorrow|today|tonight)\s*(?:at\s*)?(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))?\b",
            r"\b(morning|afternoon|evening|night)\b",
            r"\b(\d{1,2})\s*(?:o'?clock)\b"
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["time"] = match.group(0)
                break

    # 4. Enhanced topic extraction
    doc = nlp(text)

    # Remove time-related words and action words
    stop_words = {"remind", "reminder", "set", "create", "at", "on", "by", "tomorrow",
                  "today", "am", "pm", "about", "for", "to", "me", "us"}

    # Extract meaningful nouns and noun phrases
    topic_candidates = []

    # Method 1: Noun chunks
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if not any(stop_word in chunk_text for stop_word in stop_words):
            topic_candidates.append(chunk.text)

    # Method 2: Direct object extraction
    for token in doc:
        if token.dep_ in ["dobj", "pobj"] and token.pos_ in ["NOUN", "PROPN"]:
            if token.text.lower() not in stop_words:
                topic_candidates.append(token.text)

    # Method 3: Named entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "EVENT", "PRODUCT"]:
            topic_candidates.append(ent.text)

    # Select best topic
    if topic_candidates:
        result["topic"] = max(topic_candidates, key=len)  # Choose longest/most specific
    else:
        # Fallback: extract from remaining words
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if filtered_words:
            result["topic"] = " ".join(filtered_words[:5])

    # 5. Priority extraction
    priority_indicators = {
        "high": ["urgent", "important", "critical", "asap", "immediately", "high priority"],
        "medium": ["normal", "medium", "regular", "standard"],
        "low": ["low", "minor", "when possible", "low priority"]
    }

    text_lower = text.lower()
    for priority, indicators in priority_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            result["priority"] = priority
            break

    return result


def extract_project_allocation_info(text):
    """
    Advanced project allocation information extraction
    """
    result = {
        "project_name": None,
        "project_lead": None,
        "team_members": [],
        "description": None,
        "deadline": None,
        "priority": None,
        "department": None
    }

    doc = nlp(text)

    # 1. Named Entity Recognition
    persons = []
    organizations = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            organizations.append(ent.text)
        elif ent.label_ == "DATE":
            result["deadline"] = ent.text

    # Assign first person as lead, rest as team members
    if persons:
        result["project_lead"] = persons[0]
        if len(persons) > 1:
            result["team_members"] = persons[1:]

    # 2. Project name extraction
    if organizations:
        result["project_name"] = organizations[0]
    else:
        # Look for quoted strings or capitalized sequences
        quoted_match = re.search(r'"([^"]+)"', text)
        if quoted_match:
            result["project_name"] = quoted_match.group(1)
        else:
            # Extract proper nouns that might be project names
            proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
            if proper_nouns:
                result["project_name"] = " ".join(proper_nouns[:3])

    # 3. Description extraction
    # Find descriptive adjectives and relevant nouns
    descriptive_tokens = []
    for token in doc:
        if token.pos_ in ["ADJ", "NOUN"] and token.text.lower() not in {
            result["project_name"].lower() if result["project_name"] else "",
            result["project_lead"].lower() if result["project_lead"] else ""
        }:
            descriptive_tokens.append(token.text)

    if descriptive_tokens:
        result["description"] = " ".join(descriptive_tokens)

    # 4. Department/team extraction
    department_keywords = ["engineering", "marketing", "sales", "hr", "finance", "design",
                           "development", "qa", "testing", "support", "operations"]

    text_lower = text.lower()
    for dept in department_keywords:
        if dept in text_lower:
            result["department"] = dept.title()
            break

    # 5. Priority extraction (same as reminder)
    priority_indicators = {
        "high": ["urgent", "important", "critical", "asap", "immediately", "high priority"],
        "medium": ["normal", "medium", "regular", "standard"],
        "low": ["low", "minor", "when possible", "low priority"]
    }

    for priority, indicators in priority_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            result["priority"] = priority
            break

    return result


def extract_summary_info(text):
    """
    Advanced summary information extraction
    """
    result = {
        "summary": None,
        "source_content": None,
        "summary_type": None,
        "length": None
    }

    doc = nlp(text)

    # 1. Identify what needs to be summarized
    source_indicators = ["of", "from", "about", "regarding", "concerning", "on", "for"]
    source_content = None

    for i, token in enumerate(doc):
        if token.text.lower() in source_indicators and i + 1 < len(doc):
            source_content = " ".join([t.text for t in doc[i + 1:]])
            break

    result["source_content"] = source_content

    # 2. Determine summary type
    summary_types = {
        "brief": ["brief", "short", "quick", "concise"],
        "detailed": ["detailed", "comprehensive", "thorough", "complete"],
        "bullet": ["bullet", "points", "list", "bullets"]
    }

    text_lower = text.lower()
    for stype, indicators in summary_types.items():
        if any(indicator in text_lower for indicator in indicators):
            result["summary_type"] = stype
            break

    # 3. Extract length requirements
    length_patterns = [
        r"(\d+)\s*(?:words?|sentences?|paragraphs?|lines?)",
        r"(?:in\s+)?(\d+)\s*(?:words?|sentences?)"
    ]

    for pattern in length_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["length"] = match.group(1)
            break

    # 4. Generate summary if we have content
    try:
        if source_content:
            result["summary"] = generate_summary(source_content)
        else:
            result["summary"] = generate_summary(text)
    except Exception as e:
        result["summary"] = f"Summary generation failed: {str(e)}"

    return result


def extract_meeting_info(text):
    """
    Extract meeting-specific information
    """
    result = {
        "meeting_title": None,
        "attendees": [],
        "time": None,
        "date": None,
        "location": None,
        "agenda": None,
        "duration": None
    }

    doc = nlp(text)

    # Extract attendees (persons)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            result["attendees"].append(ent.text)
        elif ent.label_ in ["TIME", "DATE"]:
            if not result["time"]:
                result["time"] = ent.text

    # Extract location
    location_patterns = [
        r"(?:in|at)\s+([A-Za-z0-9\s]+(?:room|hall|office|building))",
        r"(?:room|hall)\s+([A-Za-z0-9]+)",
        r"(?:zoom|teams|meet|skype)\s*(?:meeting|call)?"
    ]

    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["location"] = match.group(0)
            break

    # Extract duration
    duration_pattern = r"(?:for\s+)?(\d+)\s*(?:hours?|hrs?|minutes?|mins?)"
    duration_match = re.search(duration_pattern, text, re.IGNORECASE)
    if duration_match:
        result["duration"] = duration_match.group(0)

    return result


def extract_task_info(text):
    """
    Extract task-specific information
    """
    result = {
        "task_title": None,
        "assignee": None,
        "deadline": None,
        "priority": None,
        "status": None,
        "description": None
    }

    doc = nlp(text)

    # Extract assignee
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            result["assignee"] = ent.text
            break
        elif ent.label_ == "DATE":
            result["deadline"] = ent.text

    # Extract status
    status_keywords = ["pending", "in progress", "completed", "blocked", "on hold"]
    text_lower = text.lower()

    for status in status_keywords:
        if status in text_lower:
            result["status"] = status
            break

    # Extract priority (same as other modules)
    priority_indicators = {
        "high": ["urgent", "important", "critical", "asap", "immediately", "high priority"],
        "medium": ["normal", "medium", "regular", "standard"],
        "low": ["low", "minor", "when possible", "low priority"]
    }

    for priority, indicators in priority_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            result["priority"] = priority
            break

    return result


def process_input(text):
    """
    Enhanced dual-pass processing with comprehensive intent detection
    """
    if not text or not isinstance(text, str):
        return {
            "module_intent": "unknown",
            "db_intent": "create",
            "processed_text": "",
            "status": "error",
            "message": "Invalid input provided"
        }

    # Preprocess the text
    processed_text = preprocess_text(text)

    # First pass: Detect DB intent (CRUD operations)
    db_intent = detect_db_intent_first_pass(processed_text)

    # Second pass: Detect module intent with DB context filtering
    module_intent = classify_module_intent_second_pass(processed_text, db_intent)

    # Extract specific information based on module intent
    extracted_data = {}

    try:
        if module_intent == "reminder":
            extracted_data = extract_reminder_info(processed_text)
        elif module_intent == "project_allocation":
            extracted_data = extract_project_allocation_info(processed_text)
        elif module_intent == "summary":
            extracted_data = extract_summary_info(processed_text)
        elif module_intent == "meeting":
            extracted_data = extract_meeting_info(processed_text)
        elif module_intent == "task":
            extracted_data = extract_task_info(processed_text)
        else:
            extracted_data = {
                "raw_content": processed_text,
                "confidence": "low"
            }
    except Exception as e:
        extracted_data = {
            "error": f"Extraction failed: {str(e)}",
            "raw_content": processed_text
        }

    # Combine all results
    result = {
        "module_intent": module_intent,
        "db_intent": db_intent,
        "processed_text": processed_text,
        "original_text": text,
        "timestamp": datetime.now().isoformat(),
        **extracted_data
    }

    return append_validation_message(result)


def append_validation_message(result: dict) -> dict:
    """
    Comprehensive validation with context-aware messaging
    """
    # Required fields mapping
    required_fields = {
        "reminder": ["time", "topic"],
        "project_allocation": ["project_name", "project_lead"],
        "summary": ["summary"],
        "meeting": ["time", "attendees"],
        "task": ["task_title"]
    }

    module_intent = result.get("module_intent")
    db_intent = result.get("db_intent")
    missing_fields = []

    # Check required fields based on module intent
    for field in required_fields.get(module_intent, []):
        value = result.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field)

    # Additional validation based on DB intent
    if db_intent in ["update", "delete"] and module_intent != "unknown":
        # For update/delete operations, we need identifiers
        identifier_fields = ["id", "name", "title", "project_name", "topic"]
        has_identifier = any(result.get(field) for field in identifier_fields)

        if not has_identifier:
            missing_fields.append("identifier")

    # Validation status and messaging
    if not missing_fields:
        result["status"] = "complete"
        result["message"] = f"Ready to {db_intent} {module_intent}"
        result["confidence"] = "high"
    else:
        # Generate human-readable field names
        readable_names = {
            "time": "a valid time",
            "topic": "what the reminder is about",
            "project_name": "the project name",
            "project_lead": "the name of the project lead",
            "description": "a brief project description",
            "summary": "the content to summarize",
            "identifier": "an identifier (name or ID) for the item to modify",
            "attendees": "list of attendees",
            "task_title": "the task title"
        }

        missing_readable = [readable_names.get(field, field) for field in missing_fields]

        result["status"] = "incomplete"
        result["message"] = f"To {db_intent} this {module_intent}, please provide: {', '.join(missing_readable)}"
        result["confidence"] = "medium" if len(missing_fields) <= 2 else "low"
        result["missing_fields"] = missing_fields

    return result
