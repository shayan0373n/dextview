import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_event(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    finger_forces = np.array(data['finger_forces'])
    finger_labels = data['finger_labels']
    trigger_index = data.get('trigger_index')
    
    # Find index for 'R Index'
    try:
        r_index_idx = finger_labels.index('R Index')
    except ValueError:
        print("Error: 'R Index' not found in finger_labels")
        return

    r_index_data = finger_forces[:, r_index_idx]
    timestamps = np.array(data['timestamps'])
    
    # Calculate relative time in ms (trigger is at 0ms)
    # If timestamps are not provided, we might need a default sampling rate, 
    # but based on the file content, they are there.
    if len(timestamps) > 1:
        sample_period = (timestamps[1] - timestamps[0]) * 1000  # ms
    else:
        sample_period = 1.953125  # Default for 512Hz if only one sample
        
    relative_time_ms = (np.arange(len(r_index_data)) - trigger_index) * sample_period
    
    plt.figure(figsize=(12, 6))
    plt.plot(relative_time_ms, r_index_data, label='R Index Force', color='blue')
    
    if trigger_index is not None:
        plt.axvline(x=0, color='red', linestyle='--', label='Trigger (t=0ms)')
    
    plt.xlabel('Time (ms) relative to trigger')
    plt.ylabel('% MVC')
    plt.title(f'R Index Force - {os.path.basename(json_path)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    session_dir = r"C:\Users\shaya\Projects\dexterity\pyquattrocento\session_20260430_191420"
    
    # Get all event_*.json files
    json_files = [f for f in os.listdir(session_dir) if f.startswith('event_') and f.endswith('.json')]
    json_files.sort()
    
    if not json_files:
        print(f"No event JSON files found in {session_dir}")
    else:
        print(f"Found {len(json_files)} events. Plotting...")
        for filename in json_files:
            json_path = os.path.join(session_dir, filename)
            output_path = filename.replace('.json', '_plot.png')
            plot_event(json_path, output_path)
        print("Batch plotting complete.")
