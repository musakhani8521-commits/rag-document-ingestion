"""Simple in-memory conversation history for follow-up questions."""


class ChatSession:
    def __init__(self, assistant, max_history=12):
        self.assistant = assistant
        self.max_history = max_history
        self.history = []

    def ask(self, question):
        result = self.assistant.ask(question, history=self.history)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        self.history = self.history[-self.max_history:]
        return result

    def clear(self):
        self.history.clear()
