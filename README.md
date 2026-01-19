# 🎯 Intelligent Video-Based Object Detection System (YOLOv8)

## 📌 Project Overview
This project implements an intelligent object detection system using deep learning and computer vision techniques. The system detects and monitors objects from video streams in real time while maintaining original video playback speed and quality. It demonstrates the practical use of YOLOv8 with Python and OpenCV for accurate video analysis.

## 🧠 Key Features
- Real-time object detection from video input  
- Maintains original FPS without speed mismatch  
- Accurate bounding box visualization  
- Supports recorded video and live camera input  
- Efficient and scalable deep learning pipeline  

## 🎯 Objectives
- Develop an automated object detection system  
- Reduce manual video monitoring  
- Apply deep learning for high-accuracy detection  
- Preserve original video playback speed  
- Provide a reusable detection framework  

## 🛠️ Technologies Used
**Software & Frameworks**
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Matplotlib  

**Hardware (Optional)**
- PC / Laptop  
- Webcam or video input  
- GPU (recommended for faster training)  

## 📂 Project Structure
📁 project-root
│── 📁 dataset
│ ├── images
│ ├── labels
│── 📁 models
│ └── best.pt
│── detect.py
│── train.py
│── demo_video.mp4
│── requirements.txt
│── README.md


## 📊 Dataset Preparation
- Images annotated in YOLO format  
- Dataset split into training and validation sets  
- Organized directory structure for efficient processing  

## 🤖 Model Training
- YOLOv8 model trained on the prepared dataset  
- Multiple epochs used to improve detection accuracy  
- Best-performing weights saved for inference  

## 🎥 Video Processing Workflow
1. Load video using OpenCV  
2. Extract frames sequentially  
3. Perform YOLOv8 inference on each frame  
4. Detect and classify objects  
5. Draw bounding boxes  
6. Display output with original FPS maintained  

## ⚙️ Installation & Setup
**Python version:** 3.9 or above

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
pip install pillow

(Optional – GPU support)

pip install torch torchvision torchaudio

▶️ How to Run

Run object detection

python detect.py


Train the model (optional)

python train.py

✅ Results

Accurate real-time object detection

Smooth video playback without frame drops

Reliable performance under varying conditions

📌 Applications

Industrial monitoring

Surveillance systems

Smart warehouses

Automated inspection

Computer vision research

🏁 Conclusion

This project presents an effective deep learning–based approach for real-time video object detection. By integrating YOLOv8 with OpenCV, the system ensures accurate detection while preserving original video quality and performance.
