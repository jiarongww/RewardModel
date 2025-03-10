import matplotlib.pyplot as plt

import json
import os
import numpy as np


# Paths to raw score JSON files
score_types = ["Aesthetic", "BLIP", "CLIP", "hps_v2", "hps_v2.1", "imagereward", "pickscore"]
output_dir = "./results_combined/"
os.makedirs(output_dir, exist_ok=True)
# Load raw scores from JSON files
raw_scores = {}
for score_type in score_types:
    file_path = f"./{score_type}/raw_scores.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            raw_scores[score_type] = json.load(f)
    else:
        print(f"File not found: {file_path}")

# Check loaded raw_scores
print("Loaded raw scores:", list(raw_scores.keys()))

# Prepare data for a grouped boxplot where each model has a set of 5 sub-columns for score types
models = list(raw_scores["Aesthetic"].keys())  # Get model IDs
score_types = list(raw_scores.keys())  # Get score types
num_score_types = len(score_types)

# Align scores using max and min for each score type
aligned_scores = {model: {score_type: [] for score_type in score_types} for model in models}

for score_type, data in raw_scores.items():
    # Find global min and max for the current score type
    all_scores = [score for model_data in data.values() for style_scores in model_data.values() for score in
                  style_scores]
    score_min, score_max = min(all_scores), max(all_scores)

    # Normalize scores for each model and style
    for model_id, model_data in data.items():
        for style, scores in model_data.items():
            normalized_scores = [(score - score_min) / (score_max - score_min) for score in scores]
            aligned_scores[model_id][score_type].extend(normalized_scores)

# Prepare data for grouped boxplot
grouped_data = []
group_labels = []
positions = []

# Parameters for grouped boxplot
group_width = 0.8  # Total width of a group of sub-columns (for a model)
box_width = group_width / num_score_types
offsets = np.linspace(-group_width / 2, group_width / 2, num_score_types)

for i, model in enumerate(models):
    for j, score_type in enumerate(score_types):
        grouped_data.append(aligned_scores[model][score_type])
        positions.append(i + 1 + offsets[j])
        if i == 0:  # Add labels only once for score types
            group_labels.append(score_type)

# Adjust colors to alternate between neighbors for clarity
neighbor_colors = plt.cm.Paired(np.linspace(0, 1, len(models)))  # Use alternating colors for clarity

plt.figure(figsize=(14, 8))

# Draw grouped boxplot
for i, model in enumerate(models):
    for j, score_type in enumerate(score_types):
        data = aligned_scores[model][score_type]
        pos = i + 1 + offsets[j]
        # Boxplot with adjusted whisker length and outlier display
        plt.boxplot(data, positions=[pos], widths=box_width, patch_artist=True,
                    whis=1.5,  # Set whisker length
                    boxprops=dict(facecolor=neighbor_colors[(3 * i) % (len(models) // 2)], color='black'),
                    medianprops=dict(color='black'),
                    flierprops=dict(marker='.', color='gray', alpha=0.05))
                    # showfliers = False)

# Add model labels at the center of each group
plt.xticks(range(1, len(models) + 1), models, rotation=45, ha='right')
plt.xlabel('Models')
plt.ylabel('Aligned Scores')
plt.title(f'Grouped Boxplot for {score_types}')
plt.grid(axis='y', linestyle='--', alpha=0.7)


# Save the plot
neighbor_colored_plot_path = os.path.join(output_dir, "group_result.png")
plt.savefig(neighbor_colored_plot_path)
plt.close()

print(f"Grouped boxplot with alternating colors for neighbors saved to {neighbor_colored_plot_path}")
