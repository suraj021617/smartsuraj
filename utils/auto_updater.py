"""
AUTO-UPDATER - Watches CSV file and auto-retrains when new data arrives
"""
import os
import time
import json
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CSVWatcher(FileSystemEventHandler):
    def __init__(self, csv_path, callback):
        self.csv_path = csv_path
        self.callback = callback
        self.last_modified = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0
        
    def on_modified(self, event):
        if event.src_path.endswith('4d_results_history.csv'):
            current_modified = os.path.getmtime(self.csv_path)
            if current_modified > self.last_modified:
                self.last_modified = current_modified
                print(f"🔄 CSV updated! Auto-retraining models...")
                self.callback()

class AutoUpdater:
    def __init__(self):
        self.update_log_file = "auto_update_log.json"
        self.load_log()
        
    def load_log(self):
        if os.path.exists(self.update_log_file):
            with open(self.update_log_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {
                'last_update': None,
                'update_count': 0,
                'updates': []
            }
    
    def save_log(self):
        with open(self.update_log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def record_update(self, rows_added=0):
        """Record an auto-update event"""
        update_entry = {
            'timestamp': datetime.now().isoformat(),
            'rows_added': rows_added,
            'status': 'success'
        }
        
        self.log['last_update'] = update_entry['timestamp']
        self.log['update_count'] += 1
        self.log['updates'].append(update_entry)
        
        # Keep only last 50 updates
        if len(self.log['updates']) > 50:
            self.log['updates'] = self.log['updates'][-50:]
        
        self.save_log()
    
    def get_update_stats(self):
        """Get statistics about auto-updates"""
        if not self.log['updates']:
            return {
                'total_updates': 0,
                'last_update': 'Never',
                'avg_rows_per_update': 0
            }
        
        total_rows = sum(u['rows_added'] for u in self.log['updates'])
        avg_rows = total_rows / len(self.log['updates']) if self.log['updates'] else 0
        
        return {
            'total_updates': self.log['update_count'],
            'last_update': self.log['last_update'],
            'avg_rows_per_update': round(avg_rows, 1),
            'recent_updates': self.log['updates'][-10:]
        }
    
    def start_watching(self, csv_path, callback):
        """Start watching CSV file for changes"""
        event_handler = CSVWatcher(csv_path, callback)
        observer = Observer()
        observer.schedule(event_handler, path=os.path.dirname(csv_path), recursive=False)
        observer.start()
        
        print(f"👁️ Watching {csv_path} for changes...")
        return observer

def check_for_new_data(df_old, df_new):
    """Check if new data was added"""
    if df_old is None or df_new is None:
        return 0
    
    old_count = len(df_old)
    new_count = len(df_new)
    
    return max(0, new_count - old_count)
