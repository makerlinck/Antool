import sys

sys.path.insert(0, "src")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import numpy as np
import tensorflow as tf

# 加载模型
model_path = "resources/models/v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

# 准备数据
print("Preparing data...")
batch_1 = np.random.rand(1, 512, 512, 3).astype(np.float32)
batch_16 = np.random.rand(16, 512, 512, 3).astype(np.float32)

# Warmup
model.predict(batch_1, verbose=0)

# Test batch=1 (16 times)
t0 = time.perf_counter()
for _ in range(16):
    model.predict(batch_1, verbose=0)
t1 = time.perf_counter()
single_total = (t1 - t0) * 1000

# Test batch=16 (1 time)
t0 = time.perf_counter()
model.predict(batch_16, verbose=0)
t1 = time.perf_counter()
batch_total = (t1 - t0) * 1000

print(f"16x single: {single_total:.1f}ms total, {single_total/16:.1f}ms/image")
print(f"1x batch16: {batch_total:.1f}ms total, {batch_total/16:.1f}ms/image")
print(f"Speedup: {single_total/batch_total:.1f}x")
