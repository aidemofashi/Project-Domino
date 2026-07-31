class Pipeline:
    def __init__(self, state, modules, make_memory=16):
        self.state = state
        self.modules = modules
        self.make_memory = make_memory
        self.chat = []
        self.messages = []

    def load_history(self, memory):
        full_chat = memory.chat_worker(chat=None)
        self.messages = list(full_chat)
        return self.messages

    def _compact_memory(self, memory):
        if len(self.chat) >= self.make_memory:
            full_chat = memory.chat_worker(self.chat)
            self.messages = list(full_chat)
            self.chat = []
