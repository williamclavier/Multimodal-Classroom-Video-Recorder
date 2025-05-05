import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
from pathlib import Path

def load_json_data(file_path: str) -> Dict:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_multimodal_analysis(output_dir: Path):
    """Create a multi-panel visualization of all model results."""
    # Load all model results
    analysis_data = load_json_data(output_dir / "analysis" / "analysis_results.json")
    ocr_data = load_json_data(output_dir / "ocr" / "ocr_results.json")
    pose_data = load_json_data(output_dir / "pose" / "pose_results.json")
    trans_data = load_json_data(output_dir / "transcription" / "transcription.json")
    decisions_data = load_json_data(output_dir / "decisions" / "decisions.json")

    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    
    # Find the maximum timestamp across all models
    max_time = max(
        max(d['timestamp'] for d in analysis_data['results']),
        max(d['timestamp'] for d in ocr_data['results']),
        max(d['timestamp'] for d in pose_data['results']),
        max(s['end'] for s in trans_data['segments']),
        max(d['timestamp'] for d in decisions_data['decisions'])
    )
    
    # Plot 1: OCR Results
    ax = axes[0]
    if len(ocr_data['results']) == 1:
        # Single value case - show horizontal line
        ocr_confidence = ocr_data['results'][0]['confidence'] / 100.0
        ax.axhline(y=ocr_confidence, color='green', alpha=1.0, linewidth=2, 
                   label=f'OCR Confidence: {ocr_confidence:.2f}')
    else:
        # Multiple values case - show line plot
        timestamps = [d['timestamp'] for d in ocr_data['results']]
        confidences = [d['confidence'] / 100.0 for d in ocr_data['results']]
        ax.plot(timestamps, confidences, label='OCR Confidence', color='green',
                alpha=1.0, linewidth=2)
    
    ax.set_title('OCR Analysis', pad=10)
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_time)  # Set consistent x-axis range
    ax.legend()
    
    # Plot 2: Transcription
    ax = axes[1]
    timestamps = [s['start'] for s in trans_data['segments']]
    confidences = [1 - s['no_speech_prob'] for s in trans_data['segments']]
    ax.plot(timestamps, confidences, label='Transcription Confidence', color='purple',
            alpha=1.0, linewidth=2)
    ax.set_title('Transcription Analysis', pad=10)
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)  # Set consistent x-axis range
    
    # Plot 3: Content Analysis
    ax = axes[2]
    timestamps = [d['timestamp'] for d in analysis_data['results']]
    confidences = [d['confidence'] for d in analysis_data['results']]
    ax.plot(timestamps, confidences, label='Content Confidence', color='blue', 
            alpha=1.0, linewidth=2)
    ax.set_title('Content Analysis', pad=10)
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)  # Set consistent x-axis range
    
    # Plot 4: Pose Results
    ax = axes[3]
    timestamps = [d['timestamp'] for d in pose_data['results']]
    confidences = [d['confidence'] for d in pose_data['results']]
    ax.plot(timestamps, confidences, label='Pose Confidence', color='red',
            alpha=1.0, linewidth=2)
    ax.set_title('Pose Analysis', pad=10)
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)  # Set consistent x-axis range
    
    # Plot 5: Final Decisions
    ax = axes[4]
    timestamps = [d['timestamp'] for d in decisions_data['decisions']]
    confidences = [d['confidence'] for d in decisions_data['decisions']]
    primary_feeds = [d['primary_feed'] for d in decisions_data['decisions']]
    
    ax.plot(timestamps, confidences, label='Decision Confidence', color='orange',
            alpha=1.0, linewidth=2)
    
    # Add transition lines and labels with improved placement
    transition_indices = [i for i in range(1, len(primary_feeds)) 
                         if primary_feeds[i] != primary_feeds[i-1]]
    
    for idx in transition_indices:
        # Add vertical line at transition point
        ax.axvline(x=timestamps[idx], color='red', linestyle='--', alpha=0.5)
        
        # Calculate y position for label (alternate between upper and lower positions)
        y_pos = 0.8 if len(transition_indices) > 0 and transition_indices.index(idx) % 2 == 0 else 0.6
        
        # Add label with transition information
        ax.annotate(
            f"{primary_feeds[idx-1]} → {primary_feeds[idx]}",
            xy=(timestamps[idx], y_pos),
            xytext=(5, 0),  # 5 points horizontal offset
            textcoords='offset points',
            ha='left',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
            rotation=0  # Horizontal text
        )
    
    ax.set_title('Final Decisions', pad=10)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)  # Set consistent x-axis range
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / "multimodal_analysis.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ Multi-model analysis plot saved to {output_path}")
    
    # Print summary statistics
    print("\nModel Confidence Statistics:")
    print(f"Content Analysis: Mean={np.mean([d['confidence'] for d in analysis_data['results']]):.3f}")
    print(f"OCR: Mean={np.mean([d['confidence']/100.0 for d in ocr_data['results']]):.3f}")
    print(f"Pose: Mean={np.mean([d['confidence'] for d in pose_data['results']]):.3f}")
    print(f"Transcription: Mean={np.mean([1 - s['no_speech_prob'] for s in trans_data['segments']]):.3f}")
    print(f"Final Decisions: Mean={np.mean([d['confidence'] for d in decisions_data['decisions']]):.3f}")

if __name__ == "__main__":
    # Get the absolute path to the output directory
    output_dir = Path(__file__).parent.parent / "output"
    
    # Check if all required files exist
    required_files = [
        output_dir / "analysis" / "analysis_results.json",
        output_dir / "ocr" / "ocr_results.json",
        output_dir / "pose" / "pose_results.json",
        output_dir / "transcription" / "transcription.json",
        output_dir / "decisions" / "decisions.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"- {f}")
        exit(1)
    
    # Run the analysis
    plot_multimodal_analysis(output_dir) 