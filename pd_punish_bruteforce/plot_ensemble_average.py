import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np # Still useful for averaging
import datetime

# --- 1. Configuration ---
results_dir = 'ensemble_results'  # Directory containing the JSON log files
num_files_to_process = None # Set to an integer to limit files, or None to process all .json files

# --- 生成唯一的输出文件名 ---
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y%m%d%H%M")
# 使用不同的变量名！ 例如 output_plot_filename
output_plot_filename = f"log{timestamp_str}.png"
print(f"Plot will be saved as: {output_plot_filename}") # 打印确认信息

# --- 2. Data Structures for Aggregation ---
step_data = defaultdict(lambda: {'cooperation': [], 'defect': [], 'punishment': []})
step_indices_set = set()
processed_files_count = 0

# --- 3. Find and Process JSON Files ---
print(f"Looking for JSON files in '{results_dir}'...")
try:
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    print(f"Found {len(all_files)} JSON files.")

    files_to_read = all_files
    if num_files_to_process is not None and num_files_to_process < len(all_files):
        files_to_read = all_files[:num_files_to_process]
        print(f"Processing the first {num_files_to_process} files.")
    else:
        print(f"Processing all {len(all_files)} files.")

    # 'filename' 在这里用作循环变量是安全的，因为它不会影响 'output_plot_filename'
    for filename_loop_var in files_to_read: # (可选) 可以给循环变量换个名字增加清晰度，但不是必须的
        filepath = os.path.join(results_dir, filename_loop_var)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            steps = data.get('steps')
            if not steps:
                print(f"Warning: Skipping '{filename_loop_var}' - no 'steps' data found.")
                continue

            for step_info in steps:
                step_index = step_info.get('step_index')
                env_data = step_info.get('environment', {})

                if step_index is not None:
                    step_indices_set.add(step_index)
                    step_data[step_index]['cooperation'].append(env_data.get('cooperation', 0))
                    step_data[step_index]['defect'].append(env_data.get('defect', 0))
                    step_data[step_index]['punishment'].append(env_data.get('punishment', 0))
                else:
                    print(f"Warning: Skipping step without 'step_index' in '{filename_loop_var}'.")

            processed_files_count += 1

        except json.JSONDecodeError:
            print(f"Warning: Skipping '{filename_loop_var}' - Invalid JSON format.")
        except Exception as e:
            print(f"Warning: Skipping '{filename_loop_var}' due to unexpected error: {e}")

except FileNotFoundError:
    print(f"Error: Directory '{results_dir}' not found.")
    exit()
except Exception as e:
    print(f"An error occurred while listing files: {e}")
    exit()


# --- 4. Calculate Averages ---
if processed_files_count == 0:
    print("Error: No valid JSON files were processed. Cannot generate plot.")
    exit()

print(f"\nSuccessfully processed {processed_files_count} files.")

sorted_step_indices = sorted(list(step_indices_set))

avg_cooperation = []
avg_defect = []
avg_punishment = []

print("Calculating averages...")
for step_idx in sorted_step_indices:
    # Calculate mean for each category at this step index
    avg_cooperation.append(np.mean(step_data[step_idx]['cooperation']) if step_data[step_idx]['cooperation'] else 0)
    avg_defect.append(np.mean(step_data[step_idx]['defect']) if step_data[step_idx]['defect'] else 0)
    avg_punishment.append(np.mean(step_data[step_idx]['punishment']) if step_data[step_idx]['punishment'] else 0)


# --- 5. Use Matplotlib 绘图 (Style from reference script) ---
print("Generating plot...")
plt.figure(figsize=(12, 6)) # 创建一个图形窗口，设置大小

# Plot the *average* values using the specified styles
plt.plot(sorted_step_indices, avg_cooperation, label='Average Cooperation', marker='o', linestyle='-')
plt.plot(sorted_step_indices, avg_defect, label='Average Defect', marker='s', linestyle='--')
plt.plot(sorted_step_indices, avg_punishment, label='Average Punishment', marker='^', linestyle=':')

# --- 6. 添加图表元素 (Style from reference script) ---
plt.title(f'Average Simulation Environment Variables Over Steps (n={processed_files_count} Runs)') # 图表标题
plt.xlabel('Step Index') # X轴标签
plt.ylabel('Average Count / Value') # Y轴标签
plt.legend() # 显示图例 (根据 plot 中的 label)
plt.grid(True) # 显示网格

# --- 7. 保存图表 ---
plt.tight_layout() # 调整布局以防止标签重叠

# 使用正确的变量名保存
plt.savefig(output_plot_filename)
plt.show() # 显示图表


