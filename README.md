# NLP Intent Engine with Session Management

A modular NLP engine using spaCy to extract structured information from natural language commands like creating reminders, summarizing content, or allocating projects. Supports Create, Read, Update, and Delete actions, and tracks user input and actions via in-memory sessions (ready for future DB logging).

---

## 🔧 Features

- Intent classification: `reminder`, `summary`, `project_allocation`
- Action recognition: `create`, `read`, `update`, `delete`
- Field extraction with validation messaging
- Session tracking with similarity replacement
- Easy extensibility for new intents
- Action session for DB-ready logging

---

