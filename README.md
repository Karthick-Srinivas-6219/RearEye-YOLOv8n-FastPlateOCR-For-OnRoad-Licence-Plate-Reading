# RearEye-YOLOv8n-FastPlateOCR-For-OnRoad-Licence-Plate-Reading
This repo is a successful attempt at replicating the licence plate reading systems that are mounted upon police patrol vehicles. It employs a 2 stage pipeline - Firstly, a YOLOv8n instance is finetuned to localize licence plates on road scenes which is fed downstream to the FastPlateOCR API which reads the digits on the plate.
