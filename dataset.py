import matplotlib.pyplot as plt

# -----------------------------
# DATASET VALUES (your project)
# -----------------------------
train = 8000
val = 932
test = 932   # update if different

labels = ['Train', 'Validation', 'Test']
values = [train, val, test]

# 🎨 Colors
colors = ['#4CAF50', '#2196F3', '#FF5722']

# -----------------------------
# PIE CHART (BEST FOR REPORT)
# -----------------------------
plt.figure(figsize=(6,6))

plt.pie(values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        shadow=True)

plt.title("Dataset Split Distribution", fontsize=14, fontweight='bold')

plt.savefig("dataset_split_pie.png", dpi=300, bbox_inches='tight')
plt.show()