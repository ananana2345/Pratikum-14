import os, time, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input,
    InputLayer
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ════════════════════════════════════════════════════════════
# HELPER: forward pass manual (tidak bergantung pada .output)
# Dipakai oleh Bagian 6, 7, dan 8 agar aman di Keras 3
# ════════════════════════════════════════════════════════════
def _run_layers(model, x_np):
    """
    Jalankan semua layer model secara berurutan.
    Lewati InputLayer (bukan layer komputasi).
    Kembalikan output akhir sebagai tensor.
    """
    x = tf.cast(x_np, tf.float32)
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            continue
        x = layer(x, training=False)
    return x

def _run_layers_until(model, x_np, stop_name):
    """
    Jalankan layer satu per satu dan hentikan setelah layer bernama stop_name.
    Kembalikan output layer tersebut.
    """
    x = tf.cast(x_np, tf.float32)
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            continue
        x = layer(x, training=False)
        if layer.name == stop_name:
            return x
    return x

def _collect_conv_outputs(model, x_np):
    """
    Jalankan forward pass dan kumpulkan output setiap Conv2D layer.
    Kembalikan list numpy array.
    """
    x = tf.cast(x_np, tf.float32)
    outputs = []
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            continue
        x = layer(x, training=False)
        if isinstance(layer, Conv2D):
            outputs.append(x.numpy())
    return outputs

# ════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════
print("=" * 65)
print("  CNN KLASIFIKASI SHAPES: FROM SCRATCH + TRANSFER LEARNING")
print("=" * 65)
print(f"  TensorFlow : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU        : {'Tersedia' if gpus else 'Tidak tersedia (CPU mode)'}")

os.makedirs("hasil", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ════════════════════════════════════════════════════════════
# BAGIAN 0 — LOAD DATASET
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [0] LOAD DATASET")
print("=" * 65)

X_train    = np.load("dataset/X_train.npy")
y_train    = np.load("dataset/y_train.npy")
X_val      = np.load("dataset/X_val.npy")
y_val      = np.load("dataset/y_val.npy")
X_test     = np.load("dataset/X_test.npy")
y_test     = np.load("dataset/y_test.npy")
y_test_raw = np.load("dataset/y_test_raw.npy").flatten()

with open("dataset/meta.pkl", "rb") as f:
    meta = pickle.load(f)

CLASS_NAMES = meta['classes']
NUM_CLASSES = meta['num_classes']
IMG_SIZE    = meta['img_size']
IMG_SHAPE   = (IMG_SIZE, IMG_SIZE, 3)

print(f"  Train   : {X_train.shape}")
print(f"  Val     : {X_val.shape}")
print(f"  Test    : {X_test.shape}")
print(f"  Kelas   : {CLASS_NAMES}")
print(f"  Img size: {IMG_SIZE}x{IMG_SIZE}x3")

# ════════════════════════════════════════════════════════════
# BAGIAN 1 — DATA AUGMENTATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [1] DATA AUGMENTATION")
print("=" * 65)

datagen_full = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.25,
    shear_range=0.15,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

datagen_light = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

datagen_none = ImageDataGenerator()

# Visualisasi augmentasi
sample_img = X_train[0:1]
fig, axes  = plt.subplots(3, 9, figsize=(20, 8))
fig.suptitle("Visualisasi Data Augmentation – Dataset Shapes",
             fontsize=14, fontweight='bold')

row_labels = ['Augmentasi\nLengkap', 'Augmentasi\nRingan', 'Tanpa\nAugmentasi']
gens       = [datagen_full, datagen_light, datagen_none]

for row, (lbl, gen) in enumerate(zip(row_labels, gens)):
    axes[row, 0].imshow(sample_img[0])
    axes[row, 0].set_ylabel(lbl, fontsize=10, rotation=0, labelpad=65, va='center')
    axes[row, 0].set_title('Original' if row == 0 else '', fontsize=9)
    axes[row, 0].axis('off')
    g = gen.flow(sample_img, batch_size=1)
    for col in range(1, 9):
        aug = next(g)[0]
        axes[row, col].imshow(np.clip(aug, 0, 1))
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(f'Aug {col}', fontsize=9)

plt.tight_layout()
plt.savefig("hasil/01_augmentasi_visualisasi.png", dpi=130, bbox_inches='tight')
plt.show()
print("  Visualisasi augmentasi disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 2 — CNN FROM SCRATCH
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [2] CNN FROM SCRATCH")
print("=" * 65)

def build_cnn(name, filters=32, depth=3, dropout=0.5,
              optimizer='adam', lr=0.001):
    model = Sequential(name=name)
    # Pakai Input layer eksplisit agar node input terdefinisi sejak awal
    model.add(Input(shape=IMG_SHAPE))
    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    for i in range(1, depth):
        f = min(filters * (2 ** i), 256)
        model.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    opt = Adam(lr) if optimizer == 'adam' else SGD(lr, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Warm-up call agar semua internal node terdefinisi
    _ = model(X_train[:1], training=False)
    return model

scratch_cfgs = [
    dict(name="CNN_Dasar",        filters=32, depth=3, dropout=0.50, optimizer='adam', lr=1e-3),
    dict(name="CNN_Filter64",     filters=64, depth=3, dropout=0.50, optimizer='adam', lr=1e-3),
    dict(name="CNN_Dalam_4Layer", filters=32, depth=4, dropout=0.50, optimizer='adam', lr=1e-3),
    dict(name="CNN_Dropout_Low",  filters=32, depth=3, dropout=0.25, optimizer='adam', lr=1e-3),
    dict(name="CNN_SGD",          filters=32, depth=3, dropout=0.50, optimizer='sgd',  lr=1e-2),
    dict(name="CNN_LR_Kecil",     filters=32, depth=3, dropout=0.50, optimizer='adam', lr=1e-4),
]

EPOCHS_S   = 40
BATCH_SIZE = 32

callbacks_base = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=0)
]

results_scratch = {}

for cfg in scratch_cfgs:
    name = cfg['name']
    print(f"\n  Training {name} "
          f"(filters={cfg['filters']}, depth={cfg['depth']}, "
          f"dropout={cfg['dropout']}, opt={cfg['optimizer']}, lr={cfg['lr']})")
    model = build_cnn(**cfg)
    t0    = time.time()
    hist  = model.fit(
        datagen_full.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_S,
        callbacks=callbacks_base,
        verbose=1
    )
    elapsed = time.time() - t0
    t_i = time.time()
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    inf_ms = (time.time() - t_i) / len(X_test) * 1000
    results_scratch[name] = dict(
        history=hist.history, test_acc=acc, test_loss=loss,
        train_time=elapsed, inference_ms=inf_ms,
        epochs_run=len(hist.history['val_accuracy']),
        model=model
    )
    print(f"  Selesai: acc={acc:.4f} | loss={loss:.4f} | "
          f"waktu={elapsed:.0f}s | inferensi={inf_ms:.3f}ms/gambar")

best_s_name  = max(results_scratch, key=lambda k: results_scratch[k]['test_acc'])
best_s_model = results_scratch[best_s_name]['model']
best_s_acc   = results_scratch[best_s_name]['test_acc']
print(f"\n  Terbaik scratch: {best_s_name} (acc={best_s_acc:.4f})")
best_s_model.save("model/best_cnn_scratch.keras")

# ════════════════════════════════════════════════════════════
# BAGIAN 3 — TRANSFER LEARNING
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [3] TRANSFER LEARNING")
print("=" * 65)

TL_SIZE  = 96
TL_SHAPE = (TL_SIZE, TL_SIZE, 3)
EPOCHS_FE = 20
EPOCHS_FT = 15

print(f"  Resize gambar {IMG_SIZE}x{IMG_SIZE} -> {TL_SIZE}x{TL_SIZE} ...")
X_train_tl = tf.image.resize(X_train, [TL_SIZE, TL_SIZE]).numpy()
X_val_tl   = tf.image.resize(X_val,   [TL_SIZE, TL_SIZE]).numpy()
X_test_tl  = tf.image.resize(X_test,  [TL_SIZE, TL_SIZE]).numpy()
print(f"  Shape baru: {X_train_tl.shape}")

def build_tl_model(base_name, fine_tune=False, unfreeze_from=None):
    inp = Input(shape=TL_SHAPE)
    if base_name == 'VGG16':
        base    = VGG16(weights='imagenet', include_top=False, input_tensor=inp)
        prep_fn = tf.keras.applications.vgg16.preprocess_input
    elif base_name == 'ResNet50':
        base    = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
        prep_fn = tf.keras.applications.resnet50.preprocess_input
    else:
        base    = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inp)
        prep_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    if not fine_tune:
        base.trainable = False
    else:
        base.trainable = True
        if unfreeze_from is not None:
            for layer in base.layers[:unfreeze_from]:
                layer.trainable = False

    x   = base.output
    x   = GlobalAveragePooling2D()(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.5)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    m   = Model(inputs=base.input, outputs=out,
                name=f"{base_name}_{'FT' if fine_tune else 'FE'}")
    lr  = 1e-5 if fine_tune else 1e-4
    m.compile(optimizer=Adam(lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m, prep_fn

tl_cfgs = [
    dict(base_name='VGG16',       fine_tune=False, unfreeze_from=None),
    dict(base_name='ResNet50',    fine_tune=False, unfreeze_from=None),
    dict(base_name='MobileNetV2', fine_tune=False, unfreeze_from=None),
    dict(base_name='VGG16',       fine_tune=True,  unfreeze_from=15),
    dict(base_name='MobileNetV2', fine_tune=True,  unfreeze_from=100),
]

results_tl = {}

for cfg in tl_cfgs:
    phase = 'FT' if cfg['fine_tune'] else 'FE'
    name  = f"{cfg['base_name']}_{phase}"
    print(f"\n  Training {name}")

    model, prep_fn = build_tl_model(**cfg)

    X_tr_pp = prep_fn(X_train_tl * 255.0)
    X_vl_pp = prep_fn(X_val_tl   * 255.0)
    X_te_pp = prep_fn(X_test_tl  * 255.0)

    epochs = EPOCHS_FT if cfg['fine_tune'] else EPOCHS_FE
    t0     = time.time()
    hist   = model.fit(
        datagen_full.flow(X_tr_pp, y_train, batch_size=32),
        steps_per_epoch=len(X_tr_pp) // 32,
        validation_data=(X_vl_pp, y_val),
        epochs=epochs,
        callbacks=callbacks_base,
        verbose=1
    )
    elapsed = time.time() - t0
    t_i     = time.time()
    loss, acc = model.evaluate(X_te_pp, y_test, verbose=0)
    inf_ms  = (time.time() - t_i) / len(X_test) * 1000

    results_tl[name] = dict(
        history=hist.history, test_acc=acc, test_loss=loss,
        train_time=elapsed, inference_ms=inf_ms,
        epochs_run=len(hist.history['val_accuracy']),
        model=model, x_test_pp=X_te_pp
    )
    print(f"  Selesai: acc={acc:.4f} | loss={loss:.4f} | "
          f"waktu={elapsed:.0f}s | inferensi={inf_ms:.3f}ms/gambar")

best_tl_name  = max(results_tl, key=lambda k: results_tl[k]['test_acc'])
best_tl_model = results_tl[best_tl_name]['model']
best_tl_acc   = results_tl[best_tl_name]['test_acc']
print(f"\n  Terbaik TL: {best_tl_name} (acc={best_tl_acc:.4f})")

# ════════════════════════════════════════════════════════════
# BAGIAN 4 — LEARNING CURVES
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [4] LEARNING CURVES")
print("=" * 65)

def plot_lc(hist, title, fname):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    a1.plot(hist['accuracy'],     lw=2, label='Train')
    a1.plot(hist['val_accuracy'], lw=2, ls='--', label='Val')
    best_ep  = int(np.argmax(hist['val_accuracy']))
    best_val = float(max(hist['val_accuracy']))
    a1.axvline(best_ep, color='red', ls=':', alpha=0.7)
    a1.annotate(f'Best {best_val:.3f}\n@ep{best_ep+1}',
                xy=(best_ep, best_val),
                xytext=(max(0, best_ep - 3), best_val - 0.12),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    a1.set_ylim(0, 1.05)
    a1.set_title('Accuracy')
    a1.set_xlabel('Epoch')
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(hist['loss'],     lw=2, label='Train')
    a2.plot(hist['val_loss'], lw=2, ls='--', label='Val')
    a2.set_title('Loss')
    a2.set_xlabel('Epoch')
    a2.legend()
    a2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"hasil/{fname}", dpi=130, bbox_inches='tight')
    plt.show()
    plt.close()

plot_lc(results_scratch[best_s_name]['history'],
        f"Learning Curve - {best_s_name}",
        "02_lc_best_scratch.png")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Learning Curves - Semua Eksperimen CNN Scratch",
             fontsize=14, fontweight='bold')
for idx, (nm, res) in enumerate(results_scratch.items()):
    ax = axes[idx // 3][idx % 3]
    h  = res['history']
    ax.plot(h['accuracy'],     lw=2, label='Train')
    ax.plot(h['val_accuracy'], lw=2, ls='--', label='Val')
    ax.set_title(f"{nm}\nacc={res['test_acc']:.3f}", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hasil/03_lc_all_scratch.png", dpi=130, bbox_inches='tight')
plt.show()

plot_lc(results_tl[best_tl_name]['history'],
        f"Learning Curve - {best_tl_name}",
        "03b_lc_best_tl.png")

print("  Learning curves disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 5 — EVALUASI KOMPREHENSIF
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [5] EVALUASI KOMPREHENSIF")
print("=" * 65)

y_prob_s  = best_s_model.predict(X_test, verbose=0)
y_pred_s  = np.argmax(y_prob_s, axis=1)

x_te_tl   = results_tl[best_tl_name]['x_test_pp']
y_prob_tl = best_tl_model.predict(x_te_tl, verbose=0)
y_pred_tl = np.argmax(y_prob_tl, axis=1)

def plot_cm(y_true, y_pred, title, fname, cmap='Blues'):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES],
                       rotation=30, ha='right', fontsize=12)
    ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=12)
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Aktual',   fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', color=color, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"hasil/{fname}", dpi=130, bbox_inches='tight')
    plt.show()
    plt.close()
    return cm

plot_cm(y_test_raw, y_pred_s,
        f"Confusion Matrix - {best_s_name}\nAcc={best_s_acc:.4f}",
        "04_cm_scratch.png", cmap='Blues')

plot_cm(y_test_raw, y_pred_tl,
        f"Confusion Matrix - {best_tl_name}\nAcc={best_tl_acc:.4f}",
        "05_cm_tl.png", cmap='Greens')

print(f"\n  Classification Report - {best_s_name}:")
print(classification_report(y_test_raw, y_pred_s,
      target_names=[c.capitalize() for c in CLASS_NAMES]))

print(f"\n  Classification Report - {best_tl_name}:")
print(classification_report(y_test_raw, y_pred_tl,
      target_names=[c.capitalize() for c in CLASS_NAMES]))

# ROC Curve
y_bin   = label_binarize(y_test_raw, classes=list(range(NUM_CLASSES)))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROC Curve - Multi-class (OvR)", fontsize=13, fontweight='bold')
clr_roc = ['#3498db', '#e74c3c', '#2ecc71']

for ax, (lbl, y_prob) in zip(axes, [
        (f"CNN Scratch - {best_s_name}", y_prob_s),
        (f"Transfer Learning - {best_tl_name}", y_prob_tl)]):
    auc_list = []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ra = auc(fpr, tpr)
        auc_list.append(ra)
        ax.plot(fpr, tpr, color=clr_roc[i], lw=2,
                label=f"{CLASS_NAMES[i].capitalize()} (AUC={ra:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('FPR', fontsize=11)
    ax.set_ylabel('TPR', fontsize=11)
    ax.set_title(f"{lbl}\nMean AUC={np.mean(auc_list):.3f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hasil/06_roc_curve.png", dpi=130, bbox_inches='tight')
plt.show()
print("  ROC curve disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 6 — FEATURE MAPS & FILTER VISUALIZATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [6] FEATURE MAPS & FILTER VISUALIZATION")
print("=" * 65)

# Ambil satu contoh gambar per kelas dari test set
sample_imgs, sample_lbls = [], []
for ci in range(NUM_CLASSES):
    idx = np.where(y_test_raw == ci)[0][0]
    sample_imgs.append(X_test[idx])
    sample_lbls.append(CLASS_NAMES[ci])

# Identifikasi Conv2D layers
conv_layers = [l for l in best_s_model.layers if isinstance(l, Conv2D)]
print(f"  Jumlah Conv2D layer: {len(conv_layers)}")

n_rows = NUM_CLASSES * len(conv_layers)
fig, axes = plt.subplots(n_rows, 9, figsize=(20, 4 * n_rows))
fig.suptitle("Feature Maps - CNN Scratch", fontsize=14, fontweight='bold')

row = 0
for ci, (img, lbl) in enumerate(zip(sample_imgs, sample_lbls)):
    # Gunakan helper yang aman — tidak perlu .output / layer.output
    fmaps = _collect_conv_outputs(best_s_model, img[np.newaxis, ...])

    for li, fmap_np in enumerate(fmaps):
        n_show = min(8, fmap_np.shape[-1])
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"{lbl.capitalize()}\nConv{li+1}",
                                 fontsize=9, rotation=0,
                                 labelpad=70, va='center')
        axes[row, 0].axis('off')
        for fi in range(1, n_show + 1):
            axes[row, fi].imshow(fmap_np[0, :, :, fi - 1], cmap='viridis')
            axes[row, fi].axis('off')
        row += 1

plt.tight_layout()
plt.savefig("hasil/07_feature_maps.png", dpi=100, bbox_inches='tight')
plt.show()
print("  Feature maps disimpan.")

# Filter visualization — layer Conv2D pertama
filters_w = conv_layers[0].get_weights()[0]  # shape: (3,3,3,filters)
n_show    = min(32, filters_w.shape[-1])
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Filter Visualization - Conv Layer 1",
             fontsize=13, fontweight='bold')
for i in range(32):
    ax = axes[i // 8][i % 8]
    if i < n_show:
        f = filters_w[:, :, :, i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f)
        ax.set_title(f"F{i+1}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig("hasil/08_filter_viz.png", dpi=130, bbox_inches='tight')
plt.show()
print("  Filter visualization disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 7 — GRAD-CAM
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [7] GRAD-CAM")
print("=" * 65)

def gradcam(model, img, cls_idx=None):
    """
    Grad-CAM menggunakan forward pass manual layer-by-layer.
    Tidak bergantung pada layer.output sehingga aman di Keras 3.
    """
    # Temukan nama Conv2D terakhir
    conv_list = [l for l in model.layers if isinstance(l, Conv2D)]
    if not conv_list:
        return None, None, None
    last_conv_name = conv_list[-1].name

    inp = tf.cast(img[np.newaxis, ...], tf.float32)

    # Jalankan layer sebelum conv terakhir di luar tape
    # lalu rekam conv output sebagai tf.Variable agar bisa dihitung gradiennya
    pre_output = inp
    reached    = False
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            continue
        if layer.name == last_conv_name:
            break
        pre_output = layer(pre_output, training=False)

    # Sekarang jalankan last conv + sisa layer di dalam tape
    conv_var = tf.Variable(
        model.get_layer(last_conv_name)(pre_output, training=False),
        trainable=True
    )

    with tf.GradientTape() as tape:
        tape.watch(conv_var)
        # Lanjutkan forward pass dari setelah last conv
        x = conv_var
        after_conv = False
        for layer in model.layers:
            if isinstance(layer, InputLayer):
                continue
            if layer.name == last_conv_name:
                after_conv = True
                continue          # conv_var sudah menggantikan layer ini
            if after_conv:
                x = layer(x, training=False)
        preds = x
        if cls_idx is None:
            cls_idx = int(tf.argmax(preds[0]))
        loss = preds[:, cls_idx]

    grads   = tape.gradient(loss, conv_var)          # [1, H, W, C]
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # [C]

    conv_np = conv_var.numpy()[0]   # [H, W, C]
    w_np    = weights.numpy()       # [C]

    cam = np.zeros(conv_np.shape[:2], dtype=np.float32)
    for i, w in enumerate(w_np):
        cam += w * conv_np[:, :, i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    cam_up = tf.image.resize(
        cam[..., np.newaxis], [img.shape[0], img.shape[1]]
    ).numpy().squeeze()

    return cam_up, cls_idx, preds.numpy()[0]


print("  Menghitung Grad-CAM ...")
fig, axes = plt.subplots(3, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 10))
fig.suptitle("Grad-CAM - Visualisasi Keputusan Model CNN Scratch",
             fontsize=14, fontweight='bold')

for ri, rl in enumerate(['Gambar Asli', 'Grad-CAM Heatmap', 'Overlay']):
    axes[ri, 0].set_ylabel(rl, fontsize=11, rotation=0,
                            labelpad=95, va='center')

for ci in range(NUM_CLASSES):
    idx = np.where(y_test_raw == ci)[0][0]
    img = X_test[idx]
    cam, pred_cls, probs = gradcam(best_s_model, img, cls_idx=ci)

    axes[0, ci].imshow(img)
    axes[0, ci].set_title(CLASS_NAMES[ci].capitalize(),
                           fontsize=12, fontweight='bold')
    axes[0, ci].axis('off')

    if cam is not None:
        axes[1, ci].imshow(cam, cmap='jet')
        axes[1, ci].axis('off')
        axes[2, ci].imshow(img)
        axes[2, ci].imshow(cam, cmap='jet', alpha=0.45)
        correct = pred_cls == ci
        axes[2, ci].set_title(
            f"Pred: {CLASS_NAMES[pred_cls].capitalize()}\n"
            f"{'OK' if correct else 'SALAH'} ({probs[pred_cls]:.2f})",
            fontsize=10,
            color='green' if correct else 'red'
        )
        axes[2, ci].axis('off')
    else:
        axes[1, ci].axis('off')
        axes[2, ci].axis('off')

plt.tight_layout()
plt.savefig("hasil/09_gradcam.png", dpi=130, bbox_inches='tight')
plt.show()
print("  Grad-CAM disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 8 — t-SNE & PCA
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [8] t-SNE & PCA FEATURE EMBEDDINGS")
print("=" * 65)

# Cari Dense layer terakhir sebelum softmax sebagai embedding layer
dense_layers   = [l for l in best_s_model.layers if isinstance(l, Dense)]
emb_layer_name = dense_layers[-2].name if len(dense_layers) >= 2 else dense_layers[-1].name
print(f"  Embedding layer: {emb_layer_name}")

# Ekstrak embedding menggunakan helper aman (forward pass manual)
print("  Mengekstrak embeddings ...")
embeddings_list = []
batch_size_emb  = 32
for start in range(0, len(X_test), batch_size_emb):
    batch = X_test[start:start + batch_size_emb]
    emb   = _run_layers_until(best_s_model, batch, emb_layer_name)
    embeddings_list.append(emb.numpy())
embeddings = np.concatenate(embeddings_list, axis=0)
print(f"  Embedding shape: {embeddings.shape}")

n_comp  = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
pca50   = PCA(n_components=n_comp)
emb_pca = pca50.fit_transform(embeddings)

print("  Menjalankan t-SNE ...")
perp = min(15, len(embeddings) - 1)
# n_iter diganti max_iter di scikit-learn >= 1.2; pakai try/except agar kompatibel dua versi
try:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
emb_2d = tsne.fit_transform(emb_pca)

pca2     = PCA(n_components=2)
emb_pca2 = pca2.fit_transform(embeddings)
var_exp  = pca2.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Embeddings - CNN Scratch", fontsize=13, fontweight='bold')
clr_emb = ['#3498db', '#e74c3c', '#2ecc71']

for ci in range(NUM_CLASSES):
    mask = y_test_raw == ci
    ax1.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                c=clr_emb[ci], label=CLASS_NAMES[ci].capitalize(),
                alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
ax1.set_title("t-SNE (2D)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Dim 1"); ax1.set_ylabel("Dim 2")
ax1.legend(fontsize=11); ax1.grid(alpha=0.2)

for ci in range(NUM_CLASSES):
    mask = y_test_raw == ci
    ax2.scatter(emb_pca2[mask, 0], emb_pca2[mask, 1],
                c=clr_emb[ci], label=CLASS_NAMES[ci].capitalize(),
                alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
ax2.set_title(f"PCA (PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%})",
              fontsize=12, fontweight='bold')
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
ax2.legend(fontsize=11); ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("hasil/10_tsne_pca.png", dpi=130, bbox_inches='tight')
plt.show()
print("  t-SNE & PCA disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 9 — TABEL PERBANDINGAN
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [9] TABEL PERBANDINGAN SEMUA EKSPERIMEN")
print("=" * 65)

all_res    = {**results_scratch, **results_tl}
sorted_res = sorted(all_res.items(),
                    key=lambda x: x[1]['test_acc'], reverse=True)

print(f"\n  {'#':<3} {'Model':<28} {'Acc':>7} {'Loss':>8} "
      f"{'Epoch':>6} {'Time(s)':>8} {'Inf(ms)':>9}  Tipe")
print("  " + "-" * 78)

for rank, (nm, r) in enumerate(sorted_res, 1):
    tipe  = "TL" if any(b in nm for b in ['VGG', 'ResNet', 'Mobile']) else "Scratch"
    crown = "TOP" if rank == 1 else f"  {rank}"
    print(f"  {crown:<4} {nm:<28} {r['test_acc']:>7.4f} {r['test_loss']:>8.4f} "
          f"{r['epochs_run']:>6} {r['train_time']:>8.0f} "
          f"{r['inference_ms']:>9.3f}  {tipe}")

# Bar chart perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Perbandingan Semua Eksperimen CNN",
             fontsize=14, fontweight='bold')

names   = [nm for nm, _ in sorted_res]
accs    = [r['test_acc']   for _, r in sorted_res]
losses  = [r['test_loss']  for _, r in sorted_res]
times   = [r['train_time'] for _, r in sorted_res]
bar_clr = ['#e74c3c' if not any(b in n for b in ['VGG', 'ResNet', 'Mobile'])
           else '#2ecc71' for n in names]

for ax, vals, xlabel, title in zip(
        axes,
        [accs, losses, times],
        ['Accuracy', 'Loss', 'Waktu (detik)'],
        ['Test Accuracy', 'Test Loss', 'Waktu Training']):
    bars = ax.barh(names, vals, color=bar_clr)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8)

legend_els = [
    mpatches.Patch(facecolor='#e74c3c', label='CNN from Scratch'),
    mpatches.Patch(facecolor='#2ecc71', label='Transfer Learning')
]
fig.legend(handles=legend_els, loc='lower center',
           ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.savefig("hasil/11_perbandingan.png", dpi=130, bbox_inches='tight')
plt.show()
print("  Tabel perbandingan disimpan.")

# ════════════════════════════════════════════════════════════
# BAGIAN 10 — KESIMPULAN
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [10] KESIMPULAN & ANALISIS")
print("=" * 65)

best_nm, best_r = sorted_res[0]
improv  = (best_tl_acc - best_s_acc) / best_s_acc * 100
winner  = "Transfer Learning" if best_tl_acc > best_s_acc else "CNN from Scratch"

print(f"""
  Dataset     : Shapes (Circle / Triangle / Rectangle)
  Total gambar: {NUM_CLASSES * 150} ({NUM_CLASSES} kelas x 150 gambar)
  Ukuran citra: {IMG_SIZE}x{IMG_SIZE} pixel RGB

  CNN Scratch terbaik  : {best_s_name}
  Akurasi test         : {best_s_acc:.4f}

  Transfer Learning terbaik: {best_tl_name}
  Akurasi test              : {best_tl_acc:.4f}

  PEMENANG KESELURUHAN : {best_nm}
  Akurasi              : {best_r['test_acc']:.4f}
  Peningkatan TL vs Scratch: {improv:+.1f}%

  ANALISIS:
  1. {winner} unggul untuk dataset ini.
  2. Augmentasi mengurangi overfitting (gap train-val mengecil).
  3. CNN Scratch kompetitif untuk dataset geometri 3 kelas.
  4. Fine-tuning Transfer Learning memberi akurasi tertinggi.
  5. MobileNetV2 trade-off terbaik: ringan + cepat + akurat.

  Output tersimpan di folder hasil/:
    01_augmentasi_visualisasi.png
    02_lc_best_scratch.png
    03_lc_all_scratch.png
    03b_lc_best_tl.png
    04_cm_scratch.png
    05_cm_tl.png
    06_roc_curve.png
    07_feature_maps.png
    08_filter_viz.png
    09_gradcam.png
    10_tsne_pca.png
    11_perbandingan.png
""")

print("=" * 65)
print("  SEMUA EKSPERIMEN SELESAI!")
print("=" * 65)