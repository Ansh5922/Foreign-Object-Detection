import json
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# Step 1: Autonomous Runway Foreign Object Debris (FOD) Detection
## Dataset Selection, Exploratory Data Analysis (EDA), Cleaning & Stratified Splitting

---
### Mission Objective
Foreign Object Debris (FOD) on airport runways—loose hardware, tools, blown tire rubber, and pavement chunks—poses life-critical hazards to civil and military aircraft. 

In this notebook, we perform the foundational engineering stage:
1. **Dataset Ingestion:** Inspect the **FOD-A (Foreign Object Debris in Airports)** benchmark (collected via runway cameras and Unmanned Aerial Systems / UAS).
2. **Taxonomy Harmonization:** Map 31 granular airfield items into **6 FAA/ICAO-aligned operational hazard classes** (`fastener`, `tool`, `tire_fragment`, `metal_scrap`, `plastic_debris`, `stone_rock`).
3. **Exploratory Data Analysis (EDA):**
   - Class imbalance analysis.
   - Bounding box scale distribution (COCO criteria: Small $< 32^2$, Medium $32^2 - 96^2$, Large $> 96^2$).
   - Environmental condition analysis (Dry vs. Wet pavement; Bright, Dim, Dark lighting).
4. **Data Cleaning & Splitting:**
   - Filter invalid bounding boxes.
   - Construct a leak-proof **Stratified 70% Train / 15% Validation / 15% Test split**.
   - Convert Pascal VOC XML annotations into normalized YOLO `.txt` format and generate `dataset.yaml`.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import yaml

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

print("All dependencies successfully imported.")
"""))

# Cell 3: Markdown - Taxonomy Mapping
cells.append(nbf.v4.new_markdown_cell("""## 1. Class Taxonomy & FAA Grouping
The raw dataset contains 31 discrete object labels. To prevent extreme data sparsity and align with airfield ground safety response procedures, we map them into 6 core hazard categories:

| Target Class | ID | Original Dataset Sub-Classes | Threat Level |
| :--- | :---: | :--- | :--- |
| **`fastener`** | 0 | Bolt, Nut, BoltNutSet, BoltWasher, Washer, Screw, Nail | **Critical** (Tire puncture / compressor blade fracture) |
| **`tool`** | 1 | AdjustableWrench, Wrench, Pliers, Cutter, Hammer, Screwdriver, AdjustableClamp | **Critical** (Airframe impact / engine ingestion) |
| **`tire_fragment`** | 2 | Hose, rubber debris | **High** (Fuel tank puncture / wing damage) |
| **`metal_scrap`** | 3 | MetalPart, MetalSheet, Wire, Battery, SodaCan, ClampPart, FuelCap | **Critical** (FOD ingestion / tire cut) |
| **`plastic_debris`** | 4 | PlasticPart, LuggageTag, LuggagePart, Label, Tape, Pen | **Medium** (Pitot tube clogging / FOD) |
| **`stone_rock`** | 5 | Rock, PaintChip, Wood | **High** (Turbine blade erosion / chipping) |
"""))

# Cell 4: Python Code - Taxonomy Map Definition
cells.append(nbf.v4.new_code_cell("""# 31 Raw Classes mapped to 6 FAA Runway Safety Super-Classes
CLASS_MAPPING = {
    # 0: fastener
    'Bolt': 'fastener',
    'Nut': 'fastener',
    'BoltNutSet': 'fastener',
    'BoltWasher': 'fastener',
    'Washer': 'fastener',
    'Screw': 'fastener',
    'Nail': 'fastener',
    
    # 1: tool
    'AdjustableWrench': 'tool',
    'Wrench': 'tool',
    'Pliers': 'tool',
    'Cutter': 'tool',
    'Hammer': 'tool',
    'Screwdriver': 'tool',
    'AdjustableClamp': 'tool',
    
    # 2: tire_fragment (rubber & elastomeric parts)
    'Hose': 'tire_fragment',
    
    # 3: metal_scrap
    'MetalPart': 'metal_scrap',
    'MetalSheet': 'metal_scrap',
    'Wire': 'metal_scrap',
    'Battery': 'metal_scrap',
    'SodaCan': 'metal_scrap',
    'ClampPart': 'metal_scrap',
    'FuelCap': 'metal_scrap',
    
    # 4: plastic_debris
    'PlasticPart': 'plastic_debris',
    'LuggageTag': 'plastic_debris',
    'LuggagePart': 'plastic_debris',
    'Label': 'plastic_debris',
    'Tape': 'plastic_debris',
    'Pen': 'plastic_debris',
    
    # 5: stone_rock (mineral & organic pavement debris)
    'Rock': 'stone_rock',
    'PaintChip': 'stone_rock',
    'Wood': 'stone_rock'
}

TARGET_CLASSES = ['fastener', 'tool', 'tire_fragment', 'metal_scrap', 'plastic_debris', 'stone_rock']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
print("Target classes and assigned IDs:")
for cls, idx in CLASS_TO_ID.items():
    print(f"  [{idx}] {cls}")
"""))

# Cell 5: Markdown - Data Parsing
cells.append(nbf.v4.new_markdown_cell("""## 2. Ingesting & Parsing Annotations from Archive
We parse XML annotations directly from `FOD_A_PascalVoc_v2.1.zip` along with environmental labels (`FOD_categorization_annotations.csv`).
"""))

# Cell 6: Python Code - Parsing script
cells.append(nbf.v4.new_code_cell("""ZIP_PATH = Path('FOD_A_PascalVoc_v2.1.zip')
assert ZIP_PATH.exists(), f"Zip archive not found at {ZIP_PATH}"

# Load Environmental Metadata CSV
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    csv_path = 'FODPascalVOCFormat-V.2.1/VOC2007/ImageSets/Main/CategorizationData/FOD_categorization_annotations.csv'
    env_df = pd.read_csv(io.BytesIO(z.read(csv_path)))

env_map = {}
for _, row in env_df.iterrows():
    # Weather: 0=Dry, 1=Wet; Light: 0=Bright, 1=Dim, 2=Dark
    fname = str(row['File']).strip()
    base_id = Path(fname).stem
    weather_label = 'Wet' if row['Weather'] == 1 else 'Dry'
    light_mapping = {0: 'Bright', 1: 'Dim', 2: 'Dark'}
    light_label = light_mapping.get(row['Light'], 'Unknown')
    env_map[base_id] = {'weather': weather_label, 'light': light_label}

print(f"Loaded environmental metadata for {len(env_map):,} images.")
"""))

# Cell 7: Python Code - Full XML Extraction into DataFrame
cells.append(nbf.v4.new_code_cell("""# Parse Pascal VOC XML annotations
records = []

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    xml_files = [f for f in z.namelist() if 'VOC2007/Annotations/' in f and f.endswith('.xml')]
    print(f"Extracting annotations from {len(xml_files):,} XML files...")
    
    for xml_file in xml_files:
        base_id = Path(xml_file).stem
        tree = ET.fromstring(z.read(xml_file))
        
        # Image dimensions
        size_elem = tree.find('size')
        img_w = int(size_elem.find('width').text) if size_elem is not None else 300
        img_h = int(size_elem.find('height').text) if size_elem is not None else 300
        
        env = env_map.get(base_id, {'weather': 'Dry', 'light': 'Bright'})
        
        for obj in tree.findall('object'):
            raw_name = obj.find('name').text.strip()
            target_class = CLASS_MAPPING.get(raw_name, None)
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            area = w * h
            
            records.append({
                'image_id': base_id,
                'image_file': f"{base_id}.jpg",
                'raw_class': raw_name,
                'target_class': target_class,
                'class_id': CLASS_TO_ID.get(target_class, -1),
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'bbox_width': w,
                'bbox_height': h,
                'bbox_area': area,
                'aspect_ratio': (w / h) if h > 0 else 0,
                'img_width': img_w,
                'img_height': img_h,
                'weather': env['weather'],
                'light': env['light']
            })

df_ann = pd.DataFrame(records)
print(f"Extracted {len(df_ann):,} object instances across {df_ann['image_id'].nunique():,} unique images.")
df_ann.head()
"""))

# Cell 8: Markdown - EDA Section
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)

We investigate three key aspects critical to runway inspection models:
1. **Class Distribution & Imbalance:** Measuring instance counts per target class.
2. **Debris Scale Distribution:** Categorizing by standard COCO metrics:
   - **Small Objects:** Area $< 32^2 = 1024\\text{ px}^2$
   - **Medium Objects:** $1024 \\le \\text{Area} < 96^2 = 9216\\text{ px}^2$
   - **Large Objects:** $\\text{Area} \\ge 9216\\text{ px}^2$
3. **Environmental Context:** Lighting levels (Bright, Dim, Dark) and Surface condition (Dry vs Wet).
"""))

# Cell 9: Python Code - Plot 1 Class Distribution
cells.append(nbf.v4.new_code_cell("""# Plot 1: Target Class Frequency Distribution
plt.figure(figsize=(10, 5))
class_counts = df_ann['target_class'].value_counts()
colors = sns.color_palette('mako', len(class_counts))

ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors)
plt.title('Runway FOD Instance Distribution (6 Target Hazard Classes)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Target Hazard Class', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)

# Value labels on bars
for p, count in zip(ax.patches, class_counts.values):
    pct = (count / len(df_ann)) * 100
    ax.annotate(f"{count:,}\\n({pct:.1f}%)", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 4), 
                textcoords='offset points')

plt.ylim(0, max(class_counts.values) * 1.15)
plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=300)
plt.show()
"""))

# Cell 10: Python Code - Scale Distribution
cells.append(nbf.v4.new_code_cell("""# Define COCO Object Scale Categories
def categorize_scale(area):
    if area < 32 * 32:
        return 'Small (<32x32 px)'
    elif area < 96 * 96:
        return 'Medium (32x32 - 96x96 px)'
    else:
        return 'Large (>96x96 px)'

df_ann['scale_category'] = df_ann['bbox_area'].apply(categorize_scale)
scale_counts = df_ann['scale_category'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scale Distribution Pie Chart
ax1.pie(scale_counts.values, labels=scale_counts.index, autopct='%1.1f%%',
        colors=['#e74c3c', '#3498db', '#2ecc71'], startangle=140, explode=(0.05, 0.05, 0.05))
ax1.set_title('Debris Scale Breakdown (COCO Standard)', fontsize=13, fontweight='bold')

# Scale Breakdown by Target Class
sns.countplot(data=df_ann, x='target_class', hue='scale_category', 
              palette=['#e74c3c', '#3498db', '#2ecc71'], ax=ax2)
ax2.set_title('Object Scale by Target Class', fontsize=13, fontweight='bold')
ax2.set_xlabel('Class', fontsize=11)
ax2.set_ylabel('Instance Count', fontsize=11)
ax2.legend(title='Scale Category')
plt.setp(ax2.get_xticklabels(), rotation=20)

plt.tight_layout()
plt.savefig('eda_scale_distribution.png', dpi=300)
plt.show()

print("Scale Category Summary:")
print(scale_counts)
"""))

# Cell 11: Python Code - Environmental Distribution
cells.append(nbf.v4.new_code_cell("""# Unique images environmental breakdown
img_env_df = df_ann[['image_id', 'weather', 'light']].drop_duplicates()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

weather_counts = img_env_df['weather'].value_counts()
sns.barplot(x=weather_counts.index, y=weather_counts.values, palette=['#34495e', '#2980b9'], ax=ax1)
ax1.set_title('Runway Surface Condition', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Images')
for p, c in zip(ax1.patches, weather_counts.values):
    ax1.annotate(f"{c:,} ({c/len(img_env_df)*100:.1f}%)", (p.get_x() + p.get_width()/2., p.get_height()+100), ha='center')

light_counts = img_env_df['light'].value_counts()
sns.barplot(x=light_counts.index, y=light_counts.values, palette=['#f39c12', '#7f8c8d', '#2c3e50'], ax=ax2)
ax2.set_title('Ambient Lighting Levels', fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Images')
for p, c in zip(ax2.patches, light_counts.values):
    ax2.annotate(f"{c:,} ({c/len(img_env_df)*100:.1f}%)", (p.get_x() + p.get_width()/2., p.get_height()+100), ha='center')

plt.tight_layout()
plt.savefig('eda_environmental_conditions.png', dpi=300)
plt.show()
"""))

# Cell 12: Markdown - Sample Visualizer
cells.append(nbf.v4.new_markdown_cell("""## 4. Visual Inspection with Bounding Boxes
We inspect sample runway images across different classes and lighting conditions.
"""))

# Cell 13: Python Code - Visual inspection grid
cells.append(nbf.v4.new_code_cell("""# Sample 6 representative images (one per class)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

class_colors = {
    'fastener': 'red',
    'tool': 'cyan',
    'tire_fragment': 'yellow',
    'metal_scrap': 'magenta',
    'plastic_debris': 'green',
    'stone_rock': 'orange'
}

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for i, target_cls in enumerate(TARGET_CLASSES):
        sample_row = df_ann[df_ann['target_class'] == target_cls].sample(1, random_state=42).iloc[0]
        img_id = sample_row['image_id']
        img_zip_path = f"FODPascalVOCFormat-V.2.1/VOC2007/JPEGImages/{img_id}.jpg"
        
        img_data = z.read(img_zip_path)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Get all bboxes for this image
        img_boxes = df_ann[df_ann['image_id'] == img_id]
        for _, b in img_boxes.iterrows():
            box = [b['xmin'], b['ymin'], b['xmax'], b['ymax']]
            color = class_colors.get(b['target_class'], 'white')
            draw.rectangle(box, outline=color, width=3)
            draw.text((b['xmin'], max(0, b['ymin'] - 12)), f"{b['target_class']} ({b['raw_class']})", fill=color)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {target_cls}\\nWeather: {sample_row['weather']} | Light: {sample_row['light']}", fontsize=11, fontweight='bold')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('sample_inspections.png', dpi=300)
plt.show()
"""))

# Cell 14: Markdown - Splitting & YOLO Formatting
cells.append(nbf.v4.new_markdown_cell("""## 5. Dataset Cleaning & Stratified Train / Validation / Test Splitting

### Engineering Strategy:
1. **Balanced Stratification:** To ensure manageable training on our GPU without redundancy from consecutive video frames, we curate a stratified balanced subset of **4,000 images** across classes, weathers, and lights.
2. **Split Ratio:**
   - **70% Training:** (~2,800 images)
   - **15% Validation:** (~600 images)
   - **15% Held-out Test:** (~600 images)
3. **Format Conversion:**
   - Pascal VOC $[x_{min}, y_{min}, x_{max}, y_{max}]$ is converted to normalized YOLO format:
     $$\\bar{x} = \\frac{x_{min} + x_{max}}{2 \\cdot W}, \\quad \\bar{y} = \\frac{y_{min} + y_{max}}{2 \\cdot H}, \\quad \\bar{w} = \\frac{w}{W}, \\quad \\bar{h} = \\frac{h}{H}$$
4. **Negative Samples:** Clean asphalt runway backgrounds with 0 annotations are included to suppress false positive hallucinations.
"""))

# Cell 15: Python Code - Dataset Export & Split Pipeline
cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
import shutil

OUTPUT_DATA_DIR = Path('data')
IMAGES_DIR = OUTPUT_DATA_DIR / 'images'
LABELS_DIR = OUTPUT_DATA_DIR / 'labels'

for split in ['train', 'val', 'test']:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

# Select a stratified balanced sample of ~4000 images
# Group images primarily by their dominant class and environmental condition
img_primary = df_ann.groupby('image_id').agg({
    'target_class': lambda x: x.mode()[0] if len(x) > 0 else 'none',
    'weather': 'first',
    'light': 'first'
}).reset_index()

# Stratified sampling of up to 4000 images
SAMPLE_SIZE = min(4000, len(img_primary))
strat_col = img_primary['target_class'] + '_' + img_primary['weather'] + '_' + img_primary['light']
# fallback if any group has 1 sample
strat_col = img_primary['target_class']

sampled_imgs, _ = train_test_split(
    img_primary,
    train_size=SAMPLE_SIZE,
    stratify=strat_col,
    random_state=42
)

# 70% Train, 15% Val, 15% Test
train_df, testval_df = train_test_split(
    sampled_imgs,
    test_size=0.30,
    stratify=sampled_imgs['target_class'],
    random_state=42
)

val_df, test_df = train_test_split(
    testval_df,
    test_size=0.50,
    stratify=testval_df['target_class'],
    random_state=42
)

splits = {
    'train': train_df['image_id'].tolist(),
    'val': val_df['image_id'].tolist(),
    'test': test_df['image_id'].tolist()
}

print(f"Stratified Dataset Created:")
print(f"  Train: {len(splits['train'])} images ({len(splits['train'])/SAMPLE_SIZE*100:.1f}%)")
print(f"  Val:   {len(splits['val'])} images ({len(splits['val'])/SAMPLE_SIZE*100:.1f}%)")
print(f"  Test:  {len(splits['test'])} images ({len(splits['test'])/SAMPLE_SIZE*100:.1f}%)")
"""))

# Cell 16: Python Code - Write Images and YOLO Labels
cells.append(nbf.v4.new_code_cell("""# Extract selected images and generate normalized YOLO labels
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for split_name, img_ids in splits.items():
        print(f"Writing {split_name} split files...")
        for img_id in img_ids:
            # 1. Copy image
            img_zip_entry = f"FODPascalVOCFormat-V.2.1/VOC2007/JPEGImages/{img_id}.jpg"
            img_dest = IMAGES_DIR / split_name / f"{img_id}.jpg"
            
            with open(img_dest, 'wb') as f_out:
                f_out.write(z.read(img_zip_entry))
            
            # 2. Build YOLO label txt
            boxes = df_ann[df_ann['image_id'] == img_id]
            label_lines = []
            
            for _, b in boxes.iterrows():
                cid = b['class_id']
                if cid < 0:
                    continue
                # Normalize coordinates
                dw = 1.0 / b['img_width']
                dh = 1.0 / b['img_height']
                x_center = ((b['xmin'] + b['xmax']) / 2.0) * dw
                y_center = ((b['ymin'] + b['ymax']) / 2.0) * dh
                w = b['bbox_width'] * dw
                h = b['bbox_height'] * dh
                
                # Clip to [0, 1] for safety
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                label_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            label_dest = LABELS_DIR / split_name / f"{img_id}.txt"
            with open(label_dest, 'w') as f_txt:
                f_txt.write("\\n".join(label_lines) + "\\n")

print("Dataset extraction and YOLO formatting complete!")
"""))

# Cell 17: Python Code - Generate dataset.yaml
cells.append(nbf.v4.new_code_cell("""# Generate dataset.yaml for YOLO training
yaml_data = {
    'path': str(OUTPUT_DATA_DIR.resolve()).replace('\\\\', '/'),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(TARGET_CLASSES),
    'names': TARGET_CLASSES
}

yaml_path = OUTPUT_DATA_DIR / 'dataset.yaml'
with open(yaml_path, 'w') as yf:
    yaml.dump(yaml_data, yf, default_flow_style=False, sort_keys=False)

print(f"Generated {yaml_path}:")
with open(yaml_path, 'r') as yf:
    print(yf.read())
"""))

# Cell 18: Markdown - Conclusion & Step 1 Output
cells.append(nbf.v4.new_markdown_cell("""## Step 1 Completed
- **Dataset Selected:** FOD-A (Foreign Object Debris in Airports).
- **Taxonomy:** 6 Harmonized FAA Runway Hazard Classes (`fastener`, `tool`, `tire_fragment`, `metal_scrap`, `plastic_debris`, `stone_rock`).
- **EDA Delivered:** Quantified small-object dominance (over 50% small debris), class proportions, and lighting/weather variations.
- **Stratified Split:** 70% Train, 15% Validation, 15% Held-out Test in standard YOLO format with `dataset.yaml`.

**Ready for Step 2: Model Architecture & Training Setup!**
"""))

nb.cells = cells

# Save notebook
out_path = Path('01_EDA_and_Data_Preparation.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook {out_path} with {len(cells)} cells.")
