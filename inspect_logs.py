import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def inspect_logs(log_dir):
    # Find the event file
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    event_file = event_files[0]
    print(f"Reading {event_file}...")

    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()['scalars']
    
    print(f"\nFound {len(tags)} scalar tags.")
    
    # Group tags by category if possible (usually by prefix)
    categories = {}
    for tag in tags:
        prefix = tag.split('/')[0] if '/' in tag else 'misc'
        if prefix not in categories:
            categories[prefix] = []
        categories[prefix].append(tag)
        
    for category, cat_tags in categories.items():
        print(f"\n--- {category} ---")
        for tag in sorted(cat_tags):
            events = ea.Scalars(tag)
            if events:
                last_event = events[-1]
                print(f"{tag}: {last_event.value:.4f} (step {last_event.step})")

if __name__ == "__main__":
    log_dir = r"D:\newton\ProtoMotions\results\test_training\lightning_logs\version_0"
    inspect_logs(log_dir)

