#!/usr/bin/env python3
import torch
import onnxruntime
import numpy as np
import cv2
import time

# 加载 ONNX 模型
onnx_model = onnxruntime.InferenceSession("model.onnx")
num = -1 
inference_time =[0]
print("--0-5 1-0 2-4 3-1 4-9 5-2 6-1 7-3 8-1 9-4 for example:if num =9 the pic's num is 4")

# 准备输入数据
#show dataset
with open("./data/MNIST/raw/train-images-idx3-ubyte","rb") as f:
    file = f.read()
for i in range(8000):
    num = num+1  
    i = 16+784*num
    image1 = [int(str(item).encode('ascii'),16) for item in file[i:i+784]]
    #print(image1)
    input_data = np.array(image1,dtype=np.float32).reshape(1,1,28,28)
    image1_np = np.array(image1,dtype=np.uint8).reshape(28,28,1)
    file_name = "test_%d.jpg"%num
    #cv2.imwrite(file_name,image1_np)
    #print(input_data)
    input_name = onnx_model.get_inputs()[0].name

    # inference 
    start_time = time.time()
    output = onnx_model.run(None, {input_name: input_data})
    end_time = time.time()
    inference_time.append(end_time - start_time) 

    # 处理输出结果
    output = torch.tensor(output[0])  # 将输出转换为 PyTorch 张量
        #print(output_tensor)
    # 输出结果处理和后续操作...
    pred =np.argmax(output)
    print("------------------------The num of this pic is ",pred,"use time ",inference_time[num]*1000,"ms")

mean = (sum(inference_time) / len(inference_time))*1000
print("loop ",num+1,"times","average time",mean,"ms")





