import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.set_page_config(page_title="CNN Scratch - Maternal Health Risk", layout="wide")

# ==================================================================================
# BAGIAN A: IMPLEMENTASI CNN 1D MURNI DARI NUMPY (TANPA TENSORFLOW / PYTORCH)
# ==================================================================================


class Conv1D:
    """Layer konvolusi 1D (padding 'same', stride=1) memakai trik im2col."""

    def __init__(self, in_channels, out_channels, kernel_size, rng):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        limit = np.sqrt(2.0 / (in_channels * kernel_size))  # inisialisasi He
        self.W = rng.normal(0, limit, size=(out_channels, in_channels, kernel_size))
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None
        self._cache = None

    def _pad(self, x):
        k = self.kernel_size
        pad_left = (k - 1) // 2
        pad_right = k - 1 - pad_left
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")
        return x_padded, pad_left, pad_right

    def forward(self, x):
        batch, in_c, length = x.shape
        x_padded, pad_left, pad_right = self._pad(x)
        length_out = length

        cols = np.zeros((batch, length_out, in_c * self.kernel_size))
        for i in range(length_out):
            window = x_padded[:, :, i:i + self.kernel_size]
            cols[:, i, :] = window.reshape(batch, -1)

        W_flat = self.W.reshape(self.out_channels, -1)
        out = cols @ W_flat.T + self.b
        out = out.transpose(0, 2, 1)

        self._cache = (x_padded, cols, x.shape, pad_left, pad_right)
        return out

    def backward(self, dout):
        x_padded, cols, x_shape, pad_left, pad_right = self._cache
        batch, in_c, length = x_shape
        length_out = dout.shape[2]

        dout_t = dout.transpose(0, 2, 1)
        W_flat = self.W.reshape(self.out_channels, -1)

        dW_flat = np.zeros_like(W_flat)
        for b_idx in range(batch):
            dW_flat += dout_t[b_idx].T @ cols[b_idx]
        self.dW = dW_flat.reshape(self.W.shape) / batch
        self.db = dout_t.sum(axis=(0, 1)) / batch

        dcols = dout_t @ W_flat
        dx_padded = np.zeros_like(x_padded)
        for i in range(length_out):
            grad_window = dcols[:, i, :].reshape(batch, in_c, self.kernel_size)
            dx_padded[:, :, i:i + self.kernel_size] += grad_window

        if pad_right > 0:
            dx = dx_padded[:, :, pad_left:-pad_right]
        else:
            dx = dx_padded[:, :, pad_left:]
        return dx

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class MaxPool1D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self._argmax = None
        self._input_shape = None
        self._length_out = None

    def forward(self, x):
        batch, channels, length = x.shape
        p = self.pool_size
        length_out = length // p
        usable = length_out * p
        x_trimmed = x[:, :, :usable]
        x_reshaped = x_trimmed.reshape(batch, channels, length_out, p)

        out = x_reshaped.max(axis=3)
        argmax = x_reshaped.argmax(axis=3)

        self._argmax = argmax
        self._input_shape = x.shape
        self._length_out = length_out
        return out

    def backward(self, dout):
        batch, channels, length = self._input_shape
        p = self.pool_size
        length_out = self._length_out

        dx = np.zeros((batch, channels, length_out, p))
        b_idx, c_idx, l_idx = np.meshgrid(
            np.arange(batch), np.arange(channels), np.arange(length_out), indexing="ij"
        )
        dx[b_idx, c_idx, l_idx, self._argmax] = dout
        dx = dx.reshape(batch, channels, length_out * p)

        full_dx = np.zeros((batch, channels, length))
        full_dx[:, :, :length_out * p] = dx
        return full_dx


class Flatten:
    def __init__(self):
        self._shape = None

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dense:
    def __init__(self, in_features, out_features, rng):
        limit = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        batch = self._x.shape[0]
        self.dW = self._x.T @ dout / batch
        self.db = dout.sum(axis=0) / batch
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self._mask = None
        self.training = True

    def forward(self, x):
        if self.training and self.rate > 0:
            self._mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self._mask
        self._mask = np.ones_like(x)
        return x

    def backward(self, dout):
        return dout * self._mask


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y_true_onehot):
    n = probs.shape[0]
    eps = 1e-9
    return -np.sum(y_true_onehot * np.log(probs + eps)) / n


def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


class Adam:
    """Optimizer Adam yang diimplementasikan manual (tanpa library deep learning)."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, "params_and_grads"):
                continue
            for p_idx, (param, grad) in enumerate(layer.params_and_grads()):
                key = (layer_idx, p_idx)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CNN1DScratch:
    def __init__(self, n_features, num_classes, f1=32, f2=64, dropout_rate=0.3, seed=42):
        rng = np.random.default_rng(seed)

        self.conv1 = Conv1D(1, f1, kernel_size=2, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool1D(pool_size=2)

        self.conv2 = Conv1D(f1, f2, kernel_size=2, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool1D(pool_size=2)

        self.flatten = Flatten()

        flat_len = max(n_features // 2 // 2, 1)
        self.fc1 = Dense(f2 * flat_len, 64, rng=rng)
        self.relu3 = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Dense(64, num_classes, rng=rng)

        self.layers_with_params = [self.conv1, self.conv2, self.fc1, self.fc2]

    def set_training(self, mode: bool):
        self.dropout.training = mode

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.dropout.forward(x)
        logits = self.fc2.forward(x)
        return logits

    def backward(self, dlogits):
        d = self.fc2.backward(dlogits)
        d = self.dropout.backward(d)
        d = self.relu3.backward(d)
        d = self.fc1.backward(d)
        d = self.flatten.backward(d)

        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d)

        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d)
        return d

    def predict_proba(self, x):
        self.set_training(False)
        logits = self.forward(x)
        return softmax(logits)

    def get_weights(self):
        return {
            "conv1_W": self.conv1.W, "conv1_b": self.conv1.b,
            "conv2_W": self.conv2.W, "conv2_b": self.conv2.b,
            "fc1_W": self.fc1.W, "fc1_b": self.fc1.b,
            "fc2_W": self.fc2.W, "fc2_b": self.fc2.b,
        }

    def set_weights(self, weights):
        self.conv1.W, self.conv1.b = weights["conv1_W"], weights["conv1_b"]
        self.conv2.W, self.conv2.b = weights["conv2_W"], weights["conv2_b"]
        self.fc1.W, self.fc1.b = weights["fc1_W"], weights["fc1_b"]
        self.fc2.W, self.fc2.b = weights["fc2_W"], weights["fc2_b"]


# ==================================================================================
# BAGIAN B: ANTARMUKA STREAMLIT
# ==================================================================================

st.title("🩺 Pelatihan Model CNN (Implementasi NumPy Murni) - Maternal Health Risk")
st.markdown("""
Aplikasi ini melatih model **1D Convolutional Neural Network (CNN)** yang diimplementasikan
**seluruhnya dari nol menggunakan NumPy** — forward pass, backpropagation, dan optimizer Adam
ditulis manual **tanpa TensorFlow maupun PyTorch**. Cocok dipakai di environment yang tidak
bisa menginstall library deep learning besar.

> Karena data ini tabular, 6 fitur (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate)
> diperlakukan sebagai "sinyal 1 dimensi" agar bisa diproses oleh layer Conv1D.
""")

# ---------------------------------------------------------
# 1. UPLOAD / LOAD DATASET
# ---------------------------------------------------------
st.header("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader("Upload file CSV (maternal_health_risk.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    with st.expander("Lihat cuplikan data"):
        st.dataframe(df.head(10))

    with st.expander("Statistik deskriptif"):
        st.dataframe(df.describe())

    with st.expander("Distribusi kelas RiskLevel"):
        fig, ax = plt.subplots()
        sns.countplot(x="RiskLevel", data=df, ax=ax, order=df["RiskLevel"].value_counts().index)
        ax.set_title("Distribusi Kelas RiskLevel")
        st.pyplot(fig)

    # -------------------------------------------------------
    # 2. PREPROCESSING
    # -------------------------------------------------------
    st.header("2️⃣ Preprocessing Data")

    target_col = "RiskLevel"
    feature_cols = [c for c in df.columns if c != target_col]

    st.write("Fitur yang digunakan:", feature_cols)
    st.write("Target:", target_col)

    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    class_names = le.classes_
    st.write("Mapping kelas:", {i: c for i, c in enumerate(class_names)})

    X = df[feature_cols].values

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Proporsi data test", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random state", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape ke (samples, channels=1, panjang_fitur)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    num_classes = len(class_names)
    y_train_oh = one_hot(y_train, num_classes)

    st.success(f"Data siap: X_train {X_train_cnn.shape}, X_test {X_test_cnn.shape}")

    # -------------------------------------------------------
    # 3. KONFIGURASI ARSITEKTUR CNN
    # -------------------------------------------------------
    st.header("3️⃣ Konfigurasi Model CNN")

    col1, col2, col3 = st.columns(3)
    with col1:
        filters_1 = st.selectbox("Jumlah filter Conv1D layer 1", [16, 32, 64], index=1)
    with col2:
        filters_2 = st.selectbox("Jumlah filter Conv1D layer 2", [32, 64, 128], index=1)
    with col3:
        dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.3, 0.05)

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Jumlah epoch", min_value=5, max_value=300, value=60, step=5)
    with col2:
        batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
    with col3:
        learning_rate = st.select_slider(
            "Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.005
        )

    n_features = X_train_cnn.shape[2]

    with st.expander("Lihat ringkasan arsitektur model"):
        temp_model = CNN1DScratch(n_features, num_classes, filters_1, filters_2, dropout_rate)
        total_params = sum(w.size for w in temp_model.get_weights().values())
        st.write(f"""
        - Conv1D(1 → {filters_1}, kernel=2) → ReLU → MaxPool1D(2)
        - Conv1D({filters_1} → {filters_2}, kernel=2) → ReLU → MaxPool1D(2)
        - Flatten → Dense(64) → ReLU → Dropout({dropout_rate})
        - Dense(64 → {num_classes}) → Softmax
        """)
        st.write(f"Total parameter: {total_params:,}")

    # -------------------------------------------------------
    # 4. PELATIHAN MODEL
    # -------------------------------------------------------
    st.header("4️⃣ Pelatihan Model")

    if st.button("🚀 Mulai Latih Model", type="primary"):
        model = CNN1DScratch(n_features, num_classes, filters_1, filters_2, dropout_rate)
        optimizer = Adam(lr=learning_rate)

        n_samples = X_train_cnn.shape[0]
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        best_val_loss = float("inf")
        best_weights = None
        patience = 15
        patience_counter = 0

        y_test_oh = one_hot(y_test, num_classes)

        for epoch in range(epochs):
            model.set_training(True)
            perm = np.random.permutation(n_samples)
            X_shuffled = X_train_cnn[perm]
            y_shuffled = y_train_oh[perm]

            epoch_loss = 0.0
            correct = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                logits = model.forward(xb)
                probs = softmax(logits)
                loss = cross_entropy_loss(probs, yb)
                epoch_loss += loss
                n_batches += 1

                preds = np.argmax(probs, axis=1)
                correct += (preds == np.argmax(yb, axis=1)).sum()

                dlogits = (probs - yb) / xb.shape[0]
                model.backward(dlogits)
                optimizer.step(model.layers_with_params)

            train_loss = epoch_loss / n_batches
            train_acc = correct / n_samples

            probs_val = model.predict_proba(X_test_cnn)
            val_loss = cross_entropy_loss(probs_val, y_test_oh)
            val_preds = np.argmax(probs_val, axis=1)
            val_acc = (val_preds == y_test).mean()

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            progress_placeholder.text(
                f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            )
            progress_bar.progress((epoch + 1) / epochs)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.copy() for k, v in model.get_weights().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    st.info(f"Early stopping pada epoch {epoch+1} (val_loss tidak membaik selama {patience} epoch)")
                    break

        if best_weights is not None:
            model.set_weights(best_weights)

        st.success("Pelatihan selesai!")

        st.session_state["model_weights"] = model.get_weights()
        st.session_state["model_config"] = (n_features, num_classes, filters_1, filters_2, dropout_rate)
        st.session_state["history"] = history
        st.session_state["X_test_cnn"] = X_test_cnn
        st.session_state["y_test"] = y_test
        st.session_state["class_names"] = class_names
        st.session_state["scaler"] = scaler
        st.session_state["label_encoder"] = le
        st.session_state["feature_cols"] = feature_cols

    # -------------------------------------------------------
    # 5. EVALUASI HASIL
    # -------------------------------------------------------
    if "model_weights" in st.session_state:
        st.header("5️⃣ Evaluasi Model")

        n_feat, n_cls, f1, f2, drop = st.session_state["model_config"]
        model = CNN1DScratch(n_feat, n_cls, f1, f2, drop)
        model.set_weights(st.session_state["model_weights"])

        history = st.session_state["history"]
        X_test_cnn = st.session_state["X_test_cnn"]
        y_test = st.session_state["y_test"]
        class_names = st.session_state["class_names"]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(history["loss"], label="Train Loss")
            ax.plot(history["val_loss"], label="Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Grafik Loss")
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.plot(history["accuracy"], label="Train Accuracy")
            ax.plot(history["val_accuracy"], label="Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Grafik Akurasi")
            ax.legend()
            st.pyplot(fig)

        probs_test = model.predict_proba(X_test_cnn)
        y_pred = np.argmax(probs_test, axis=1)

        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi pada Data Test", f"{acc*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")
            report = classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())

        # -----------------------------------------------------
        # 6. SIMPAN MODEL & PREDIKSI DATA BARU
        # -----------------------------------------------------
        st.header("6️⃣ Simpan Model & Uji Prediksi Baru")

        model_path = "cnn_scratch_maternal_health_model.pkl"
        save_payload = {
            "weights": model.get_weights(),
            "config": st.session_state["model_config"],
            "scaler": st.session_state["scaler"],
            "label_encoder": st.session_state["label_encoder"],
            "feature_cols": st.session_state["feature_cols"],
        }
        with open(model_path, "wb") as f:
            pickle.dump(save_payload, f)
        with open(model_path, "rb") as f:
            st.download_button(
                "⬇️ Download Model (.pkl)", f, file_name=model_path, mime="application/octet-stream"
            )

        st.subheader("Coba Prediksi Data Baru")
        feature_cols = st.session_state["feature_cols"]
        scaler = st.session_state["scaler"]

        input_vals = []
        cols = st.columns(len(feature_cols))
        default_vals = {"Age": 30, "SystolicBP": 120, "DiastolicBP": 80,
                         "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76}
        for i, fc in enumerate(feature_cols):
            with cols[i]:
                val = st.number_input(fc, value=float(default_vals.get(fc, 0)))
                input_vals.append(val)

        if st.button("Prediksi"):
            new_data = np.array(input_vals).reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            new_data_cnn = new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])

            pred_prob = model.predict_proba(new_data_cnn)[0]
            pred_class = class_names[np.argmax(pred_prob)]
            st.success(f"Prediksi Risiko: **{pred_class.upper()}**")
            st.bar_chart(pd.Series(pred_prob, index=class_names))

else:
    st.info("Silakan upload file `maternal_health_risk.csv` terlebih dahulu untuk memulai.")