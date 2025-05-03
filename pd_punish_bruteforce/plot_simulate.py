import json
import matplotlib.pyplot as plt

# --- 1. 定义 JSON 文件名 ---
json_filename = 'log202505040005.json' # 使用你实际的文件名

# --- 2. 读取和解析 JSON 数据 ---
try:
    with open(json_filename, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：文件 '{json_filename}' 未找到。请确保文件在当前目录下或提供正确路径。")
    exit()
except json.JSONDecodeError:
    print(f"错误：文件 '{json_filename}' 格式无效，无法解析为 JSON。")
    exit()

# --- 3. 提取绘图所需数据 ---
steps_data = data.get('steps', []) # 使用 .get() 避免 KeyError

step_indices = []
cooperation_values = []
defect_values = []
punishment_values = []

# 检查 steps_data 是否为空
if not steps_data:
    print(f"错误：在 '{json_filename}' 中未找到 'steps' 数据或 'steps' 列表为空。")
    exit()
    
for step in steps_data:
    step_index = step.get('step_index')
    environment_data = step.get('environment', {}) # 使用 .get() 获取环境字典

    if step_index is not None: # 确保 step_index 存在
        step_indices.append(step_index)
        # 使用 .get() 并提供默认值 0，以防某个变量在某一步缺失
        cooperation_values.append(environment_data.get('cooperation', 0))
        defect_values.append(environment_data.get('defect', 0))
        punishment_values.append(environment_data.get('punishment', 0))
    else:
         print(f"警告：跳过一个缺少 'step_index' 的步骤。")

# 检查是否成功提取了数据
if not step_indices:
    print(f"错误：未能从 '{json_filename}' 中提取有效的步骤数据进行绘图。")
    exit()

# --- 4. 使用 Matplotlib 绘图 ---
plt.figure(figsize=(12, 6)) # 创建一个图形窗口，设置大小

plt.plot(step_indices, cooperation_values, label='Cooperation', marker='o', linestyle='-')
plt.plot(step_indices, defect_values, label='Defect', marker='s', linestyle='--')
plt.plot(step_indices, punishment_values, label='Punishment', marker='^', linestyle=':')

# --- 5. 添加图表元素 ---
plt.title('Simulation Environment Variables Over Steps') # 图表标题
plt.xlabel('Step Index') # X轴标签
plt.ylabel('Count / Value') # Y轴标签
plt.legend() # 显示图例 (根据 plot 中的 label)
plt.grid(True) # 显示网格

# --- 6. 显示或保存图表 ---
plt.tight_layout() # 调整布局以防止标签重叠
plt.show() # 显示图表

# 如果你想保存图表而不是显示，可以取消下面这行的注释，并注释掉 plt.show()
# plt.savefig('simulation_plot.png') 
# print("图表已保存为 simulation_plot.png")
