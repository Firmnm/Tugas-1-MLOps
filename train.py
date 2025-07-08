/*
* 225150207111017 Farhan Rahmansyah
* 225150200111001 Firman Maulana
* 225150200111009 Lucas Chandra
* 225150207111002 Shofy Pramesti Putri Guruh
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
import os
import warnings

# Import skops untuk menyimpan model yang kompatibel dengan Hugging Face Spaces
import skops.io as sio
from skops.io import get_untrusted_types

warnings.filterwarnings("ignore")


class PersonalityClassifier:
    def __init__(self, data_path="Data/personality_datasert.csv"):
        """
        Inisialisasi PersonalityClassifier

        Args:
            data_path (str): Path ke dataset
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_and_explore_data(self):
        """Memuat dan eksplorasi dataset"""
        print("Memuat dataset...")
        self.data = pd.read_csv(self.data_path)

        print(f"Ukuran dataset: {self.data.shape}")
        print("\nInfo dataset:")
        print(self.data.info())

        print("\nBeberapa baris pertama:")
        print(self.data.head())

        print("\nDistribusi target:")
        print(self.data["Personality"].value_counts())

        print("\nNilai yang hilang:")
        print(self.data.isnull().sum())

        print("\nStatistik dasar:")
        print(self.data.describe())

        return self.data

    def preprocess_data(self):
        """Preprocessing data untuk training"""
        print("\nPreprocessing data...")

        # Menangani nilai yang hilang jika ada
        self.data = self.data.dropna()

        # Encoding variabel kategorikal
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].map({"Yes": 1, "No": 0})

        # Memisahkan fitur dan target
        X = self.data.drop("Personality", axis=1)
        y = self.data["Personality"]

        # Encoding variabel target
        y_encoded = self.label_encoder.fit_transform(y)

        # Membagi data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Ukuran training set: {self.X_train.shape}")
        print(f"Ukuran test set: {self.X_test.shape}")
        print(f"Distribusi target training: {np.bincount(self.y_train)}")
        print(f"Distribusi target test: {np.bincount(self.y_test)}")

    def visualize_data(self):
        """Membuat visualisasi untuk eksplorasi data"""
        print("\nMembuat visualisasi...")

        # Membuat figure dengan subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Eksplorasi Dataset Personality", fontsize=16)

        # Distribusi variabel target
        self.data["Personality"].value_counts().plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Distribusi Personality")
        axes[0, 0].set_xlabel("Tipe Personality")
        axes[0, 0].set_ylabel("Jumlah")

        # Heatmap korelasi
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=axes[0, 1])
            axes[0, 1].set_title("Heatmap Korelasi Fitur")

        # Distribusi fitur
        features_to_plot = [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
        ]
        for i, feature in enumerate(features_to_plot):
            if feature in self.data.columns:
                row, col = divmod(i + 2, 3)
                if row < 2:
                    sns.boxplot(
                        data=self.data, x="Personality", y=feature, ax=axes[row, col]
                    )
                    axes[row, col].set_title(f"{feature} berdasarkan Personality")

        plt.tight_layout()
        plt.savefig("Results/data_exploration.png", dpi=300, bbox_inches="tight")

    def train_best_model(self):
        """Melatih model terbaik dengan hyperparameter tuning"""
        print("\nMelatih model Random Forest dengan hyperparameter tuning...")

        # Menggunakan Random Forest sebagai model terbaik
        # berdasarkan performa yang konsisten untuk klasifikasi personality
        best_model = RandomForestClassifier(random_state=42)

        # Parameter terbaik untuk Random Forest
        best_params = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }

        # Membuat pipeline dengan scaling
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", best_model)]  # Normalisasi fitur
        )

        # Melakukan grid search untuk menemukan parameter terbaik
        print("Mencari parameter terbaik...")
        grid_search = GridSearchCV(
            pipeline,
            best_params,
            cv=5,  # 5-fold cross validation
            scoring="accuracy",  # Menggunakan akurasi sebagai metrik
            n_jobs=-1,  # Menggunakan semua CPU core
            verbose=1,  # Menampilkan progress
        )

        # Melatih model dengan data training
        grid_search.fit(self.X_train, self.y_train)

        # Menyimpan model terbaik
        self.best_model = grid_search.best_estimator_

        # Menampilkan hasil
        print(f"Random Forest - Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Random Forest - Best Parameters: {grid_search.best_params_}")

        # Menyimpan hasil untuk evaluasi
        model_results = {
            "Random Forest": {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "model": grid_search.best_estimator_,
            }
        }

        return model_results

    def evaluate_model(self, model_results):
        """Evaluasi model terbaik pada data test"""
        print("\nMengevaluasi model terbaik pada data test...")

        # Prediksi pada data test
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # Menghitung metrik evaluasi
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC Score: {auc_score:.4f}")

        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        # Membuat plot evaluasi
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=target_names,
            yticklabels=target_names,
        )
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Prediksi")
        axes[0].set_ylabel("Aktual")

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {auc_score:.2f})",
        )
        axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend(loc="lower right")

        # Feature Importance (untuk Random Forest)
        if hasattr(self.best_model.named_steps["model"], "feature_importances_"):
            importances = self.best_model.named_steps["model"].feature_importances_
            feature_names = self.X_train.columns
            indices = np.argsort(importances)[::-1]

            axes[2].bar(range(len(importances)), importances[indices])
            axes[2].set_title("Tingkat Kepentingan Fitur")
            axes[2].set_xlabel("Fitur")
            axes[2].set_ylabel("Tingkat Kepentingan")
            axes[2].set_xticks(range(len(importances)))
            axes[2].set_xticklabels([feature_names[i] for i in indices], rotation=45)

        plt.tight_layout()
        plt.savefig("Results/model_evaluation.png", dpi=300, bbox_inches="tight")

        print("\nHasil Model:")
        print("-" * 50)
        for name, results in model_results.items():
            print(f"{name}: CV Score = {results['best_score']:.4f}")

        # Menyimpan metrik ke file
        with open("Results/metrics.txt", "w") as outfile:
            outfile.write(
                f"\nAccuracy = {round(accuracy, 4)}, AUC Score = {round(auc_score, 4)}."
            )
            outfile.write(
                f"\nCV Score = {round(model_results['Random Forest']['best_score'], 4)}."
            )

        print("Metrik disimpan ke Results/metrics.txt")

        return accuracy, auc_score

    def save_model(self):
        """Menyimpan model yang telah dilatih menggunakan skops untuk deployment di Hugging Face Spaces"""
        print("\nMenyimpan model untuk deployment di Hugging Face Spaces...")

        # Membuat direktori Model jika belum ada
        os.makedirs("Model", exist_ok=True)

        # Menyimpan HANYA model dalam format skops untuk Hugging Face Spaces
        # File ini yang akan digunakan untuk deployment
        sio.dump(self.best_model, "Model/personality_classifier.skops")
        print("✓ Model disimpan dalam format skops untuk Hugging Face Spaces")

        # Menyimpan label encoder dalam format skops juga untuk konsistensi
        sio.dump(self.label_encoder, "Model/label_encoder.skops")
        print("✓ Label encoder disimpan dalam format skops")

        # Menyimpan nama fitur untuk validasi input
        feature_names = list(self.X_train.columns)
        sio.dump(feature_names, "Model/feature_names.skops")
        print("✓ Nama fitur disimpan dalam format skops")

        # Menampilkan untrusted types untuk keamanan deployment
        print("\nMemeriksa keamanan model untuk deployment...")
        unknown_types = get_untrusted_types(file="Model/personality_classifier.skops")
        print("Unknown types yang diperlukan:", unknown_types)

        # Test loading untuk memastikan model dapat dimuat dengan benar
        print("Menguji loading model...")
        try:
            sio.load(
                "Model/personality_classifier.skops", trusted=unknown_types
            )
            print("✓ Model berhasil dimuat dan siap untuk deployment!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

        print("\n" + "=" * 60)
        print("FILE UNTUK HUGGING FACE SPACES DEPLOYMENT:")
        print("=" * 60)
        print("📁 Model/personality_classifier.skops     <- MODEL UTAMA")
        print("📁 Model/label_encoder.skops             <- UNTUK DECODE HASIL")
        print("📁 Model/feature_names.skops             <- UNTUK VALIDASI INPUT")
        print("📁 Results/metrics.txt                   <- METRIK PERFORMA")
        print("=" * 60)
        print(
            "Gunakan file personality_classifier.skops sebagai model utama di Hugging Face Spaces"
        )

    def load_model_from_skops(self, model_path="Model/personality_classifier.skops"):
        """
        Memuat model yang disimpan menggunakan skops untuk deployment

        Args:
            model_path (str): Path ke file model .skops

        Returns:
            model: Model yang telah dimuat
        """
        print(f"Memuat model dari {model_path}...")

        # Mendapatkan untrusted types
        unknown_types = get_untrusted_types(file=model_path)
        print(f"Unknown types yang diperlukan: {unknown_types}")

        # Memuat model
        loaded_model = sio.load(model_path, trusted=unknown_types)

        print("Model berhasil dimuat!")
        return loaded_model

    def run_complete_pipeline(self):
        """Menjalankan pipeline training lengkap"""
        print("=" * 60)
        print("PELATIHAN MODEL KLASIFIKASI PERSONALITY")
        print("=" * 60)

        # Membuat direktori Results jika belum ada
        os.makedirs("Results", exist_ok=True)

        # Memuat dan eksplorasi data
        self.load_and_explore_data()

        # Preprocessing data
        self.preprocess_data()

        # Visualisasi data
        self.visualize_data()

        # Melatih model terbaik dengan hyperparameter tuning
        model_results = self.train_best_model()

        # Evaluasi model terbaik
        accuracy, auc_score = self.evaluate_model(model_results)

        # Menyimpan model
        self.save_model()

        print("\n" + "=" * 60)
        print("PELATIHAN SELESAI DENGAN SUKSES!")
        print(f"Akurasi Model Terbaik pada Test: {accuracy:.4f}")
        print(f"AUC Model Terbaik pada Test: {auc_score:.4f}")
        print("\n🚀 MODEL SIAP UNTUK DEPLOYMENT DI HUGGING FACE SPACES!")
        print("File utama: Model/personality_classifier.skops")
        print("=" * 60)

        return self.best_model, accuracy, auc_score


def main():
    """Fungsi utama untuk menjalankan pipeline training"""
    # Inisialisasi classifier
    classifier = PersonalityClassifier()

    # Menjalankan pipeline lengkap
    best_model, accuracy, auc_score = classifier.run_complete_pipeline()

    return best_model, accuracy, auc_score


if __name__ == "__main__":
    # Menjalankan pipeline training
    model, acc, auc = main()
