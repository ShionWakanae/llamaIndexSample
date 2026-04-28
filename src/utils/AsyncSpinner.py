import threading
import time
from rich.text import Text

class AsyncSpinner:
    def __init__(self):
        self.live = None
        self.running = False
        self.thread = None
    
    def _animate(self):
        chars = ["/", "-", "\\", "|"]
        i = 0
        while self.running:
            if self.live:
                char = chars[i % len(chars)]
                self.live.update(Text(f"...{char}", style="cyan"))
            i += 1
            time.sleep(0.5)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.live:
            self.live.update(Text("", style="green"))