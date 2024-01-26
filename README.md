# YOLOP-Lane-Keeping-Assist

This repository contains a script that demonstrates the application of YOLOP/YOLO in real-time applications (games), utilizing the autonomous driving-related prediction capabilities (road/lane detection) of the [YOLOP](https://github.com/hustvl/YOLOP) model to visualize and create a straightforward lane assist system that tries to maintain the vehicle inside the lanes and emergency brake when another vehicle is blocking the way at a close distance.

# Real-Time Execution

The code was tested on Euro Truck Simulator 1 in real time. TensorRT was used to improve the inference performance, but it was eventually removed since it provided significantly less accuracy compared with PyTorch in both FP16 and FP32 precision data types. The models employed the default pretrained weights; thus, the accuracy still suffers due to the sim2real gap.

https://github.com/stefanos50/YOLOP-Lane-Keeping-Assist/assets/36155283/800521bf-f3c4-40a7-b7ce-dcd61249f28b

