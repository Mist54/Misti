from datetime import datetime

from nlp.nlp_engine import process_input, nlp


# Enhanced user input management class
class nlp_userInput:
    """
    Advanced user input manager with comprehensive tracking and processing
    """

    def __init__(self):
        self._data = []
        self._processed_data = []
        self._session_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_inputs": 0,
            "unique_inputs": 0
        }

    def set(self, value: str):
        """
        Enhanced input processing with deduplication and similarity checks
        """
        if not value or not isinstance(value, str):
            return

        # Update metadata
        self._session_metadata["total_inputs"] += 1

        # Skip exact duplicate
        if value in self._data:
            return

        # Semantic similarity check with recent inputs
        if self._data:
            # Check similarity with last 3 inputs for efficiency
            recent_inputs = self._data[-3:]
            for recent_input in recent_inputs:
                try:
                    last_doc = nlp(recent_input)
                    new_doc = nlp(value)

                    # Use higher threshold for better deduplication
                    if last_doc.similarity(new_doc) > 0.85:
                        # Update the most recent similar input
                        idx = self._data.index(recent_input)
                        self._data[idx] = value

                        # Update corresponding processed data
                        processed_result = process_input(value)
                        self._processed_data[idx] = processed_result
                        return
                except Exception:
                    # If similarity check fails, continue with normal processing
                    pass

        # Add new input and process it
        self._data.append(value)
        self._session_metadata["unique_inputs"] += 1

        try:
            processed_result = process_input(value)
            self._processed_data.append(processed_result)
        except Exception as e:
            # Handle processing errors gracefully
            error_result = {
                "module_intent": "unknown",
                "db_intent": "create",
                "processed_text": value,
                "original_text": value,
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            self._processed_data.append(error_result)

    def get(self):
        """Get all raw input data"""
        return self._data.copy()

    def get_processed(self):
        """Get all processed data"""
        return self._processed_data.copy()

    def get_last_processed(self):
        """Get the most recent processed result"""
        return self._processed_data[-1] if self._processed_data else None

    def get_by_intent(self, module_intent=None, db_intent=None):
        """Get processed data filtered by intent types"""
        filtered_data = []

        for item in self._processed_data:
            match = True
            if module_intent and item.get("module_intent") != module_intent:
                match = False
            if db_intent and item.get("db_intent") != db_intent:
                match = False

            if match:
                filtered_data.append(item)

        return filtered_data

    def get_statistics(self):
        """Get session statistics"""
        stats = self._session_metadata.copy()

        # Add intent distribution
        module_intents = {}
        db_intents = {}

        for item in self._processed_data:
            module_intent = item.get("module_intent", "unknown")
            db_intent = item.get("db_intent", "create")

            module_intents[module_intent] = module_intents.get(module_intent, 0) + 1
            db_intents[db_intent] = db_intents.get(db_intent, 0) + 1

        stats["module_intent_distribution"] = module_intents
        stats["db_intent_distribution"] = db_intents
        stats["processing_success_rate"] = len([i for i in self._processed_data if i.get("status") != "error"]) / len(
            self._processed_data) if self._processed_data else 0

        return stats

    def clear(self):
        """Clear all session data"""
        self._data = []
        self._processed_data = []
        self._session_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_inputs": 0,
            "unique_inputs": 0
        }
        print("Session has been cleared.")

    def export_session(self):
        """Export session data for persistence"""
        return {
            "raw_data": self._data,
            "processed_data": self._processed_data,
            "metadata": self._session_metadata,
            "exported_at": datetime.now().isoformat()
        }


class ActionSession:
    """
    Enhanced action session with comprehensive logging and analytics
    """

    def __init__(self):
        self.actions = []
        self.session_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0
        }

    def log(self, action_data: dict):
        """Log action with enhanced metadata"""
        if not isinstance(action_data, dict):
            return

        # Enhance action data with metadata
        enhanced_action = action_data.copy()
        enhanced_action.update({
            "timestamp": datetime.now().isoformat(),
            "session_id": len(self.actions) + 1,
            "action_id": f"action_{len(self.actions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })

        self.actions.append(enhanced_action)

        # Update metadata
        self.session_metadata["total_actions"] += 1
        if enhanced_action.get("status") == "complete":
            self.session_metadata["successful_actions"] += 1
        elif enhanced_action.get("status") == "error":
            self.session_metadata["failed_actions"] += 1

    def get_logs(self):
        """Get all action logs"""
        return self.actions.copy()

    def get_logs_by_db_intent(self, db_intent):
        """Get actions filtered by DB intent"""
        return [action for action in self.actions if action.get("db_intent") == db_intent]

    def get_logs_by_module_intent(self, module_intent):
        """Get actions filtered by module intent"""
        return [action for action in self.actions if action.get("module_intent") == module_intent]

    def get_logs_by_status(self, status):
        """Get actions filtered by status"""
        return [action for action in self.actions if action.get("status") == status]

    def get_recent_logs(self, count=10):
        """Get most recent actions"""
        return self.actions[-count:] if len(self.actions) >= count else self.actions

    def get_session_analytics(self):
        """Get comprehensive session analytics"""
        if not self.actions:
            return {"message": "No actions logged yet"}

        analytics = self.session_metadata.copy()

        # Intent distribution
        module_intents = {}
        db_intents = {}
        statuses = {}

        for action in self.actions:
            # Module intent distribution
            module_intent = action.get("module_intent", "unknown")
            module_intents[module_intent] = module_intents.get(module_intent, 0) + 1

            # DB intent distribution
            db_intent = action.get("db_intent", "create")
            db_intents[db_intent] = db_intents.get(db_intent, 0) + 1

            # Status distribution
            status = action.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

        analytics.update({
            "module_intent_distribution": module_intents,
            "db_intent_distribution": db_intents,
            "status_distribution": statuses,
            "success_rate": self.session_metadata["successful_actions"] / self.session_metadata["total_actions"] if
            self.session_metadata["total_actions"] > 0 else 0,
            "most_common_module_intent": max(module_intents, key=module_intents.get) if module_intents else None,
            "most_common_db_intent": max(db_intents, key=db_intents.get) if db_intents else None
        })

        return analytics

    def search_logs(self, query=None, **filters):
        """Search and filter action logs"""
        results = self.actions

        # Apply filters
        for key, value in filters.items():
            results = [action for action in results if action.get(key) == value]

        # Apply text search if query provided
        if query:
            query_lower = query.lower()
            filtered_results = []

            for action in results:
                # Search in various text fields
                searchable_fields = ["original_text", "processed_text", "topic", "project_name", "message"]

                for field in searchable_fields:
                    field_value = action.get(field)
                    if field_value and isinstance(field_value, str) and query_lower in field_value.lower():
                        filtered_results.append(action)
                        break

            results = filtered_results

        return results

    def clear(self):
        """Clear all action logs"""
        self.actions = []
        self.session_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0
        }
        print("Action session has been cleared.")

    def export_logs(self, format_type="json"):
        """Export action logs in various formats"""
        export_data = {
            "actions": self.actions,
            "metadata": self.session_metadata,
            "exported_at": datetime.now().isoformat(),
            "total_actions": len(self.actions)
        }

        if format_type.lower() == "json":
            import json
            return json.dumps(export_data, indent=2)
        elif format_type.lower() == "csv":
            # Simplified CSV export of key fields
            import csv
            import io

            output = io.StringIO()
            if self.actions:
                fieldnames = ["timestamp", "module_intent", "db_intent", "status", "message"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

                for action in self.actions:
                    row = {field: action.get(field, "") for field in fieldnames}
                    writer.writerow(row)

            return output.getvalue()
        else:
            return export_data


# Utility functions for advanced processing
def analyze_intent_confidence(result):
    """Analyze and score the confidence of intent detection"""
    confidence_score = 0
    confidence_factors = []

    # Check if key fields are extracted
    module_intent = result.get("module_intent")
    db_intent = result.get("db_intent")

    if module_intent != "unknown":
        confidence_score += 30
        confidence_factors.append("Module intent identified")

    if db_intent != "create":  # Non-default DB intent
        confidence_score += 20
        confidence_factors.append("Specific DB intent detected")

    # Check extraction quality
    status = result.get("status")
    if status == "complete":
        confidence_score += 40
        confidence_factors.append("All required fields extracted")
    elif status == "incomplete":
        confidence_score += 20
        confidence_factors.append("Partial extraction successful")

    # Check for specific extracted data
    extracted_fields = ["time", "topic", "project_name", "project_lead", "summary", "attendees"]
    extracted_count = sum(1 for field in extracted_fields if result.get(field))

    confidence_score += min(extracted_count * 5, 30)
    if extracted_count > 0:
        confidence_factors.append(f"{extracted_count} specific fields extracted")

    # Normalize score to 0-100 range
    confidence_score = min(confidence_score, 100)

    return {
        "confidence_score": confidence_score,
        "confidence_level": "high" if confidence_score >= 80 else "medium" if confidence_score >= 50 else "low",
        "confidence_factors": confidence_factors
    }


def suggest_improvements(result):
    """Suggest improvements for incomplete or low-confidence results"""
    suggestions = []

    module_intent = result.get("module_intent")
    status = result.get("status")
    missing_fields = result.get("missing_fields", [])

    if status == "incomplete":
        if "time" in missing_fields:
            suggestions.append("Try specifying a more explicit time (e.g., '3 PM tomorrow' instead of 'later')")

        if "topic" in missing_fields:
            suggestions.append("Be more specific about what the reminder is for")

        if "project_name" in missing_fields:
            suggestions.append("Mention the project name explicitly or put it in quotes")

        if "project_lead" in missing_fields:
            suggestions.append("Include the full name of the person responsible")

    if module_intent == "unknown":
        suggestions.append("Try using more specific keywords like 'remind me', 'assign project', or 'summarize'")

    # Confidence-based suggestions
    confidence_analysis = analyze_intent_confidence(result)
    if confidence_analysis["confidence_level"] == "low":
        suggestions.append("Consider rephrasing with clearer action words and more specific details")

    return suggestions