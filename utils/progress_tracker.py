import time

class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current=None):
        if current:
            self.current = current
        else:
            self.current += 1
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = (self.total - self.current) * avg_time
            
            print(f"\rProgress: {self.current}/{self.total} ({self.current*100//self.total}%) | "
                  f"Elapsed: {int(elapsed)}s | ETA: {int(remaining)}s", end='', flush=True)
    
    def finish(self):
        print(f"\nCompleted in {int(time.time() - self.start_time)}s")
