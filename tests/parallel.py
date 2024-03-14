import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# 假设这是您的函数
def test(input_data):
    time.sleep(1)  # 假设这是某种耗时的操作
    return f"Processed {input_data}"

# 使用并行处理并显示进度的函数
def parallel_test(inputs):
    max_workers = os.cpu_count()  # 获取CPU核心数量
    print(max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test, data) for data in tqdm(inputs)]

        results = [f.result() for f in tqdm(futures, total=len(futures), desc="Processing")]

    return results

# 示例输入
inputs = list(range(1, 11))

# 并行处理并显示进度
outputs = parallel_test(inputs)

# 打印结果
for output in outputs:
    print(output)