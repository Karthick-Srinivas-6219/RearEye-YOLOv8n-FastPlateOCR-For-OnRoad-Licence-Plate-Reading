# 🚓📸 RearEye-YOLOv8n-FastPlateOCR-For-OnRoad-Licence-Plate-Reading
This repo is a successful attempt at replicating the licence plate reading systems that are mounted upon police patrol vehicles. It employs a 2 stage pipeline - Firstly, a YOLOv8n instance is finetuned to localize licence plates on road scenes which is fed downstream to the FastPlateOCR API which reads the digits on the plate.

# Demo 👇
<video src="demo.mp4" controls width="640"></video>
[[Link to Demo]](https://youtu.be/bTiBW0bzcz8 "Click to watch")

# Overview of the pipeline
![Alt text](Full_System_Architecture.png)

## 🚀 Features

* **Finetuned YOLOv8n**: Accurate licence plate localization **Precision = 0.954, Recall = 0.963, mAP@50 = 0.973**.
* **FastPlateOCR**: Another **open source project** that offers pre-trained models for **licence plate style OCR**.
---
