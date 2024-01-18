import matplotlib.pyplot as plt
import numpy as np

# Input data
datasets = ["WSN-DS", "WSNBFSF", "NSL-KDD", "CICIDS2017"]
time_all_features = [39, 40, 54.5, 103]
time_selected_features = [19, 25, 45, 83.5]
num_all_features = [18, 15, 42, 67]
num_selected_features = [9, 5, 19, 14]

# Set up positions for the bars
bar_width = 0.2
bar_positions = np.arange(len(datasets))

# Create bar graph
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(bar_positions - bar_width/2, time_all_features, bar_width, label='All Features', color='darkorange')
bar2 = ax.bar(bar_positions + bar_width/2, time_selected_features, bar_width, label='Selected Features', color='blue')

# Add data labels
for i, (all_feat, sel_feat) in enumerate(zip(time_all_features, time_selected_features)):
    ax.text(bar_positions[i] - bar_width/2, all_feat + 0.1, str(num_all_features[i]), ha='center', fontsize=12, color='black')
    ax.text(bar_positions[i] + bar_width/2, sel_feat + 0.1, str(num_selected_features[i]), ha='center', fontsize=12, color='black')

# Set axis labels and title
ax.set_xlabel('Datasets', fontsize=12)
ax.set_ylabel('Inference Time (ms)', fontsize=12)
ax.set_title('Inference Time Comparison with All Features and Selected Features', fontsize=14)
ax.set_xticks(bar_positions)
ax.set_xticklabels(datasets)
ax.legend()

# Zoom in to emphasize the difference between numbers
ax.set_ylim([0, 110])

# Show the plot
save_path=f"./figures/the relationship between testing time and PGSWO-based feature selection without hyperparameters optimization.jpg"
plt.savefig(save_path, format='jpg', bbox_inches='tight')

plt.show()