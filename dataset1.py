# ============================================================
# PROGRAM 1: MEMBUAT DATASET SENDIRI
# Dataset: 3 kelas → Shapes (Circle, Triangle, Rectangle)
# Masing-masing 150 gambar (lebih dari minimal 100)
# Simpan sebagai: 01_create_dataset.py
# ============================================================

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ── Konfigurasi ──────────────────────────────────────────────
IMG_SIZE    = 64        # 64×64 pixel
N_PER_CLASS = 150       # gambar per kelas (minimal 100)
CLASSES     = ['circle', 'triangle', 'rectangle']
NUM_CLASSES = len(CLASSES)
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

os.makedirs("dataset/raw/circle",    exist_ok=True)
os.makedirs("dataset/raw/triangle",  exist_ok=True)
os.makedirs("dataset/raw/rectangle", exist_ok=True)
os.makedirs("dataset",               exist_ok=True)

print("=" * 60)
print("  MEMBUAT DATASET SHAPES (BENTUK GEOMETRI)")
print("=" * 60)
print(f"  Kelas     : {CLASSES}")
print(f"  Per kelas : {N_PER_CLASS} gambar")
print(f"  Ukuran    : {IMG_SIZE}×{IMG_SIZE} pixel, RGB")
print(f"  Total     : {NUM_CLASSES * N_PER_CLASS} gambar\n")

# ── Helper: warna & background acak ──────────────────────────
def rand_color(min_v=50, max_v=230):
    return tuple(random.randint(min_v, max_v) for _ in range(3))

def rand_bg():
    mode = random.choice(['solid', 'gradient', 'noise'])
    img  = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)
    if mode == 'solid':
        img.paste(rand_color(180, 255), [0, 0, IMG_SIZE, IMG_SIZE])
    elif mode == 'gradient':
        c1, c2 = rand_color(160, 255), rand_color(160, 255)
        for y in range(IMG_SIZE):
            r = int(c1[0] + (c2[0]-c1[0]) * y/IMG_SIZE)
            g = int(c1[1] + (c2[1]-c1[1]) * y/IMG_SIZE)
            b = int(c1[2] + (c2[2]-c1[2]) * y/IMG_SIZE)
            draw.line([(0,y),(IMG_SIZE,y)], fill=(r,g,b))
    else:  # noise
        arr  = np.random.randint(180, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img  = Image.fromarray(arr)
    return img

def augment_img(img):
    """Augmentasi ringan agar setiap gambar unik."""
    # Rotasi acak
    angle = random.uniform(-20, 20)
    img   = img.rotate(angle, expand=False,
                        fillcolor=tuple(np.random.randint(200,255,3).tolist()))
    # Brightness & contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    # Blur ringan sesekali
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img

# ── Fungsi Gambar Bentuk ──────────────────────────────────────

def draw_circle(idx):
    img  = rand_bg()
    draw = ImageDraw.Draw(img)
    margin = random.randint(6, 16)
    x0 = random.randint(margin, IMG_SIZE//2 - margin)
    y0 = random.randint(margin, IMG_SIZE//2 - margin)
    r  = random.randint(12, IMG_SIZE//2 - margin)
    x1, y1 = min(x0+2*r, IMG_SIZE-margin), min(y0+2*r, IMG_SIZE-margin)
    fill    = rand_color(30, 200)
    outline = rand_color(0, 100)
    thickness = random.randint(1, 4)
    draw.ellipse([x0, y0, x1, y1], fill=fill,
                 outline=outline, width=thickness)
    # Kadang tambah lingkaran kecil (noise)
    if random.random() < 0.3:
        xn = random.randint(0, IMG_SIZE-8)
        yn = random.randint(0, IMG_SIZE-8)
        draw.ellipse([xn,yn,xn+6,yn+6],
                     fill=rand_color(150,220), outline=None)
    return augment_img(img)

def draw_triangle(idx):
    img  = rand_bg()
    draw = ImageDraw.Draw(img)
    cx   = random.randint(20, IMG_SIZE-20)
    cy   = random.randint(20, IMG_SIZE-20)
    r    = random.randint(15, 24)
    # Variasi bentuk segitiga: equilateral, isosceles, random
    variant = random.choice(['equilateral', 'isosceles', 'random'])
    if variant == 'equilateral':
        angles = [90, 210, 330]
    elif variant == 'isosceles':
        angles = [90, 220, 320]
    else:
        angles = sorted(random.sample(range(0, 360), 3))
    pts = [(int(cx + r * np.cos(np.radians(a))),
            int(cy - r * np.sin(np.radians(a)))) for a in angles]
    fill    = rand_color(30, 200)
    outline = rand_color(0, 100)
    draw.polygon(pts, fill=fill, outline=outline)
    return augment_img(img)

def draw_rectangle(idx):
    img  = rand_bg()
    draw = ImageDraw.Draw(img)
    margin = random.randint(6, 16)
    x0 = random.randint(margin, IMG_SIZE//2)
    y0 = random.randint(margin, IMG_SIZE//2)
    w  = random.randint(16, IMG_SIZE - x0 - margin)
    h  = random.randint(16, IMG_SIZE - y0 - margin)
    # Variasi: persegi, persegi panjang horizontal/vertikal
    if random.random() < 0.3:
        h = w  # persegi
    x1, y1 = x0 + w, y0 + h
    fill    = rand_color(30, 200)
    outline = rand_color(0, 100)
    thickness = random.randint(1, 4)
    draw.rectangle([x0, y0, x1, y1], fill=fill,
                   outline=outline, width=thickness)
    # Kadang ada bayangan
    if random.random() < 0.25:
        shadow_offset = 3
        draw.rectangle([x0+shadow_offset, y0+shadow_offset,
                         x1+shadow_offset, y1+shadow_offset],
                        outline=(100,100,100), width=1)
    return augment_img(img)

draw_fn = {
    'circle':    draw_circle,
    'triangle':  draw_triangle,
    'rectangle': draw_rectangle
}

# ── Generate & Simpan Gambar ──────────────────────────────────
for cls in CLASSES:
    print(f"  Membuat gambar kelas '{cls}'...", end=' ')
    for i in range(N_PER_CLASS):
        img  = draw_fn[cls](i)
        path = f"dataset/raw/{cls}/{cls}_{i:04d}.png"
        img.save(path)
    print(f"✅ {N_PER_CLASS} gambar disimpan → dataset/raw/{cls}/")

# ── Load & Susun Array NumPy ──────────────────────────────────
print("\n  Menyusun array dataset...")
X_all, y_all = [], []

for cls_idx, cls in enumerate(CLASSES):
    folder = f"dataset/raw/{cls}"
    files  = sorted(os.listdir(folder))
    for fname in files:
        if not fname.endswith('.png'):
            continue
        img_path = os.path.join(folder, fname)
        img      = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr      = np.array(img, dtype=np.float32) / 255.0
        X_all.append(arr)
        y_all.append(cls_idx)

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"  Shape X: {X_all.shape}   Shape y: {y_all.shape}")
print(f"  Distribusi: { {CLASSES[i]: int((y_all==i).sum()) for i in range(NUM_CLASSES)} }")

# ── Shuffle ───────────────────────────────────────────────────
perm  = np.random.permutation(len(X_all))
X_all = X_all[perm]
y_all = y_all[perm]

# ── Split: 70% train / 15% val / 15% test ─────────────────────
n        = len(X_all)
n_train  = int(n * 0.70)
n_val    = int(n * 0.15)

X_train, y_train = X_all[:n_train],        y_all[:n_train]
X_val,   y_val   = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
X_test,  y_test  = X_all[n_train+n_val:],  y_all[n_train+n_val:]

# One-hot encoding
import tensorflow as tf
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"\n  Split dataset:")
print(f"    Training   : {X_train.shape[0]} gambar")
print(f"    Validasi   : {X_val.shape[0]} gambar")
print(f"    Test       : {X_test.shape[0]} gambar")

# ── Simpan ─────────────────────────────────────────────────────
np.save("dataset/X_train.npy",      X_train)
np.save("dataset/y_train.npy",      y_train_cat)
np.save("dataset/X_val.npy",        X_val)
np.save("dataset/y_val.npy",        y_val_cat)
np.save("dataset/X_test.npy",       X_test)
np.save("dataset/y_test.npy",       y_test_cat)
np.save("dataset/y_test_raw.npy",   y_test)

with open("dataset/meta.pkl", "wb") as f:
    pickle.dump({"classes": CLASSES, "img_size": IMG_SIZE,
                 "num_classes": NUM_CLASSES}, f)

print("\n  ✅ Semua file dataset tersimpan di folder 'dataset/'")

# ── Visualisasi Sample ────────────────────────────────────────
fig, axes = plt.subplots(3, 10, figsize=(20, 7))
fig.suptitle("Dataset Shapes – Sample Gambar per Kelas",
             fontsize=15, fontweight='bold')

for cls_idx, cls in enumerate(CLASSES):
    idx_cls = np.where(y_train == cls_idx)[0][:10]
    for col, i in enumerate(idx_cls):
        ax = axes[cls_idx, col]
        ax.imshow(X_train[i])
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(cls.capitalize(), fontsize=12,
                          fontweight='bold', rotation=0,
                          labelpad=55, va='center')

plt.tight_layout()
os.makedirs("hasil", exist_ok=True)
plt.savefig("hasil/00_sample_dataset.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Distribusi kelas ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Distribusi Kelas Dataset Shapes", fontsize=13, fontweight='bold')
colors = ['#3498db', '#e74c3c', '#2ecc71']
splits = [('Training', y_train), ('Validasi', y_val), ('Test', y_test)]

for ax, (title, y) in zip(axes, splits):
    counts = [(y == i).sum() for i in range(NUM_CLASSES)]
    bars   = ax.bar(CLASSES, counts, color=colors)
    ax.set_title(f"{title} ({sum(counts)} gambar)")
    ax.set_ylabel("Jumlah")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                str(v), ha='center', fontsize=10)
    ax.set_ylim([0, max(counts)*1.2])

plt.tight_layout()
plt.savefig("hasil/00b_distribusi_kelas.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("  ✅ DATASET SELESAI DIBUAT!")
print(f"  Total: {n} gambar | {NUM_CLASSES} kelas | {IMG_SIZE}×{IMG_SIZE}px")
print("=" * 60)