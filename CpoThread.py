# my_thread.py
import threading
import time

class CpoThread(threading.Thread):
    def __init__(self, shared_value):
        threading.Thread.__init__(self)
        self.shared_value = shared_value

    def run(self):
        while True:
            # Sleep for 5 seconds
            time.sleep(10)
            # Change the value
            self.shared_value[0] = 0.4 if self.shared_value[0] == 0.2 else 0.2
            #print(f"Value changed to {self.shared_value[0]}")
