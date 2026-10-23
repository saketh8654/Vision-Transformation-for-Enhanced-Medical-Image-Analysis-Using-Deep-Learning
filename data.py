import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# CLASSES
# -----------------------------
classes = ['XRAY_Normal', 'XRAY_Diseased', 'CT_Normal', 'CT_Diseased']

# -----------------------------
# VALUES (your dataset)
# -----------------------------
train = [2000, 2000, 2000, 2000]
val   = [232, 232, 232, 232]
test  = [232, 232, 232, 232]

x = np.arange(len(classes))
width = 0.25

# 🎨 Colors
train_color = '#2980b9'
val_color   = '#f39c12'
test_color  = '#27ae60'

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10,6))

bars1 = plt.bar(x - width, train, width, label='Train', color=train_color)
bars2 = plt.bar(x, val, width, label='Validation', color=val_color)
bars3 = plt.bar(x + width, test, width, label='Test', color=test_color)

plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Dataset Distribution (Train / Validation / Test)", fontsize=14, fontweight='bold')

plt.xticks(x, classes, rotation=20)

# Add values
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + 20,
                 str(height),
                 ha='center',
                 fontsize=9,
                 fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("dataset_distribution.png", dpi=300)
plt.show()