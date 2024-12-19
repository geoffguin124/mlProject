import os
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

model_a_path = 'runs/detect/train2/weights/best.pt'
model_b_path = 'runs/detect/train7/weights/best.pt'
model_baseline_path = 'yolov8n.pt'

model_a = YOLO(model_a_path)
model_b = YOLO(model_b_path)
model_baseline = YOLO(model_baseline_path)

data_yaml_a = 'data.yaml' # 32 classes
data_yaml_b = 'test_2/data.yaml' # 4 classes

print("=== Evaluating Model A ===")
val_results_a = model_a.val(data=data_yaml_a)
print(val_results_a)

print("=== Evaluating Model B ===")
val_results_b = model_b.val(data=data_yaml_b)
print(val_results_b)

print("=== Evaluating Baseline Model ===")
val_results_baseline = model_baseline.val(data=data_yaml_b)
print(val_results_baseline)

metrics = {
    'Model': ['Model A', 'Model B', 'Baseline'],
    'mAP@0.5': [val_results_a.box.map50, val_results_b.box.map50, val_results_baseline.box.map50],
    'mAP@0.5:0.95': [val_results_a.box.map, val_results_b.box.map, val_results_baseline.box.map],
    'Precision': [val_results_a.box.mp, val_results_b.box.mp, val_results_baseline.box.mp],
    'Recall': [val_results_a.box.mr, val_results_b.box.mr, val_results_baseline.box.mr],
}

df_metrics = pd.DataFrame(metrics)
print("\n=== Summary of Model Metrics ===")
print(df_metrics)

test_images_dir = 'test_data'

image_files = []
for folder in os.listdir(test_images_dir):
    subfolder_path = os.path.join(test_images_dir, folder)
    if os.path.isdir(subfolder_path):
        images_in_subfolder = [os.path.join(subfolder_path, f)
                               for f in os.listdir(subfolder_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Selecting 1500 images to test
        images_in_subfolder = images_in_subfolder[:1500]
        image_files.extend(images_in_subfolder)

models = {
    'Model A': model_a,
    'Model B': model_b,
    'Baseline': model_baseline
}

results_data = []
all_model_inference_data = {}

for model_name, model_obj in models.items():
    detection_counts = []
    confidences_all = []
    class_counts = {}

    for img_path in image_files:
        preds = model_obj.predict(img_path, imgsz=640, verbose=False)
        result = preds[0]
        boxes = result.boxes
        confs = [b.conf.item() for b in boxes]
        cls_ids = [int(b.cls.item()) for b in boxes]

        detection_counts.append(len(boxes))
        confidences_all.extend(confs)

        for c in cls_ids:
            class_counts[c] = class_counts.get(c, 0) + 1

    avg_det = statistics.mean(detection_counts) if detection_counts else 0.0
    med_det = statistics.median(detection_counts) if detection_counts else 0.0
    std_det = statistics.pstdev(detection_counts) if len(detection_counts) > 1 else 0.0

    avg_conf = statistics.mean(confidences_all) if confidences_all else 0.0
    med_conf = statistics.median(confidences_all) if confidences_all else 0.0
    std_conf = statistics.pstdev(confidences_all) if len(confidences_all) > 1 else 0.0

    results_data.append({
        'Model': model_name,
        'Avg Detection Count': avg_det,
        'Median Detection Count': med_det,
        'Std Detection Count': std_det,
        'Avg Confidence': avg_conf,
        'Median Confidence': med_conf,
        'Std Confidence': std_conf,
        'Class Counts': class_counts
    })

    # Store for plotting later
    all_model_inference_data[model_name] = (detection_counts, confidences_all)

results_df = pd.DataFrame(results_data)
print("\n=== Inference Statistics on Unlabeled Test Images ===")
print(results_df[['Model', 'Avg Detection Count', 'Median Detection Count', 'Std Detection Count',
                  'Avg Confidence', 'Median Confidence', 'Std Confidence']])

# Histogram of detection counts
plt.figure(figsize=(10, 5))
for rd in results_data:
    model_name = rd['Model']
    detection_counts, _ = all_model_inference_data[model_name]
    plt.hist(detection_counts, bins=10, alpha=0.5, label=model_name)

plt.title("Distribution of Detection Counts per Image")
plt.xlabel("Number of Detections")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Histogram of confidence scores
plt.figure(figsize=(10, 5))
for rd in results_data:
    model_name = rd['Model']
    _, confidences_all = all_model_inference_data[model_name]
    plt.hist(confidences_all, bins=10, alpha=0.5, label=model_name)

plt.title("Distribution of Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.legend()
plt.show()
