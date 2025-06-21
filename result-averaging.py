import os
import csv
from collections import defaultdict
from tabulate import tabulate

# Directory containing the results CSV files
results_dir = "./results"
fixed_noise_level = "0.1"  # Replace with the desired noise level
seed_values = range(0, 21)  # Replace with the desired list of seed values

# Initialize data structures to store aggregated data
head_data = defaultdict(lambda: {'f1_sum': 0, 'accuracy_sum': 0, 'count': 0})

# Process each CSV file in the results directory
for file_name in os.listdir(results_dir):
    for seed in seed_values:
        if file_name.startswith(f"results_noise_{fixed_noise_level}_seed_{seed}") and file_name.endswith(".csv"):
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    head = row['Head']
                    f1_drop = float(row['F1 Drop (Contribution)'])
                    accuracy_drop = float(row['Accuracy Drop'])

                    # Aggregate data for each head
                    head_data[head]['f1_sum'] += f1_drop
                    head_data[head]['accuracy_sum'] += accuracy_drop
                    head_data[head]['count'] += 1

# Calculate mean values and prepare for reranking
averaged_heads = []
for head, data in head_data.items():
    mean_f1 = data['f1_sum'] / data['count']
    mean_accuracy = data['accuracy_sum'] / data['count']
    averaged_heads.append({
        'Head': head,
        'Mean F1 Drop': mean_f1,
        'Mean Accuracy Drop': mean_accuracy
    })

# Rerank heads based on mean F1 Drop and mean Accuracy Drop
averaged_heads.sort(key=lambda x: x['Mean Accuracy Drop'], reverse=True)

# Write the averaged and reranked results to a new CSV file
output_file_name = f"./Acc avg results/averaged_results_noise_{fixed_noise_level}.csv"
with open(output_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Rank", "Head", "Mean F1 Drop", "Mean Accuracy Drop"])
    for rank, head_info in enumerate(averaged_heads, 1):
        writer.writerow([rank, head_info['Head'], head_info['Mean F1 Drop'], head_info['Mean Accuracy Drop']])

print(f"Averaged results written to {output_file_name}")

# Print the table of averaged results
print("\nAveraged Results Table:")
print(tabulate(averaged_heads, headers="keys", showindex=True))