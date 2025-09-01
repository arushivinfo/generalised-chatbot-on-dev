
import time
import json
from pathlib import Path
TIMINGS_FILE = Path(__file__).parent / "timings.json"

# Reset file at the start of the app
if not TIMINGS_FILE.exists() or "reset_timings" not in globals():
    TIMINGS_FILE.write_text(json.dumps({}, indent=2))
    reset_timings = True

def timed(func):
    """Decorator to measure execution time of a function and store in JSON."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start

        # Load current timings
        try:
            with open(TIMINGS_FILE, "r") as f:
                timings_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            timings_data = {}

        # Save or update timing
        timings_data[func.__name__] = f"{duration:.6f} seconds"

        # Write updated JSON
        with open(TIMINGS_FILE, "w") as f:
            json.dump(timings_data, f, indent=2)

        #print(f"[TIMER] {func.__name__} took {duration:.6f} seconds")
        return result
    return wrapper


