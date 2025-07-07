import gradio as gr
import numpy as np
import pandas as pd
import skops.io as sio
from skops.io import get_untrusted_types
import os


# Configuration for server based on environment
def get_server_config():
    """Get server configuration based on environment"""
    # Check if running in Docker
    is_docker = (
        os.path.exists("/.dockerenv")
        or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
    )

    if is_docker:
        # Docker environment - use 0.0.0.0 to allow external access
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
        print(f"🐳 Running in Docker - Server: {server_name}:{server_port}")
    else:
        # Local development - use 127.0.0.1 for security
        server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
        print(f"💻 Running locally - Server: {server_name}:{server_port}")

    return server_name, server_port


class PersonalityClassifierApp:
    def __init__(self):
        """Inisialisasi aplikasi personality classifier"""
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.load_models()

    def load_models(self):
        """Memuat model dan preprocessing objects menggunakan skops"""
        try:
            print("🔄 Memuat model dan preprocessing objects...")

            # Menentukan base path untuk model files
            base_path = os.path.dirname(os.path.abspath(__file__))
            parent_path = os.path.dirname(base_path)

            # Mencoba beberapa lokasi untuk file model
            possible_paths = [
                "Model/personality_classifier.skops",  # Local development
                os.path.join(
                    parent_path, "Model/personality_classifier.skops"
                ),  # Relative to App folder
                "./Model/personality_classifier.skops",  # Current directory
                "personality_classifier.skops",  # Hugging Face Spaces root
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path:
                unknown_types = get_untrusted_types(file=model_path)
                self.model = sio.load(model_path, trusted=unknown_types)
                print(f"✅ Model berhasil dimuat dari: {model_path}")
            else:
                print(
                    f"❌ File model tidak ditemukan di lokasi manapun: {possible_paths}"
                )
                return False

            # Memuat label encoder
            encoder_possible_paths = [
                "Model/label_encoder.skops",
                os.path.join(parent_path, "Model/label_encoder.skops"),
                "./Model/label_encoder.skops",
                "label_encoder.skops",
            ]

            encoder_path = None
            for path in encoder_possible_paths:
                if os.path.exists(path):
                    encoder_path = path
                    break

            if encoder_path:
                unknown_types = get_untrusted_types(file=encoder_path)
                self.label_encoder = sio.load(encoder_path, trusted=unknown_types)
                print(f"✅ Label encoder berhasil dimuat dari: {encoder_path}")
            else:
                print(
                    f"❌ File label encoder tidak ditemukan di lokasi manapun: {encoder_possible_paths}"
                )
                return False

            # Memuat feature names
            features_possible_paths = [
                "Model/feature_names.skops",
                os.path.join(parent_path, "Model/feature_names.skops"),
                "./Model/feature_names.skops",
                "feature_names.skops",
            ]

            features_path = None
            for path in features_possible_paths:
                if os.path.exists(path):
                    features_path = path
                    break

            if features_path:
                unknown_types = get_untrusted_types(file=features_path)
                self.feature_names = sio.load(features_path, trusted=unknown_types)
                print(f"✅ Feature names berhasil dimuat dari: {features_path}")
                print(f"Features: {self.feature_names}")
            else:
                print(
                    f"❌ File feature names tidak ditemukan di lokasi manapun: {features_possible_paths}"
                )
                return False

            return True

        except Exception as e:
            print(f"❌ Error memuat model: {str(e)}")
            return False

    def predict_personality(
        self,
        time_alone,
        stage_fear,
        social_events,
        going_outside,
        drained_socializing,
        friends_circle,
        post_frequency,
    ):
        """
        Prediksi personality berdasarkan input user

        Args:
            time_alone: Waktu yang dihabiskan sendirian (jam/hari)
            stage_fear: Takut tampil di depan umum (Yes/No)
            social_events: Kehadiran acara sosial (skala 1-10)
            going_outside: Frekuensi keluar rumah (skala 1-10)
            drained_socializing: Merasa lelah setelah bersosialisasi (Yes/No)
            friends_circle: Ukuran lingkaran pertemanan
            post_frequency: Frekuensi posting di media sosial (skala 1-10)

        Returns:
            tuple: (formatted_result, plot_data, plot_visibility)
        """
        try:
            if self.model is None:
                return "❌ Model belum dimuat dengan benar", None, False

            # Konversi input kategorical
            stage_fear_encoded = 1 if stage_fear == "Yes" else 0
            drained_encoded = 1 if drained_socializing == "Yes" else 0

            # Membuat array input sesuai urutan feature names dari dataset
            input_data = []
            for feature in self.feature_names:
                if feature == "Time_spent_Alone":
                    input_data.append(time_alone)
                elif feature == "Stage_fear":
                    input_data.append(stage_fear_encoded)
                elif feature == "Social_event_attendance":
                    input_data.append(social_events)
                elif feature == "Going_outside":
                    input_data.append(going_outside)
                elif feature == "Drained_after_socializing":
                    input_data.append(drained_encoded)
                elif feature == "Friends_circle_size":
                    input_data.append(friends_circle)
                elif feature == "Post_frequency":
                    input_data.append(post_frequency)

            # Reshape untuk prediksi
            input_array = np.array(input_data).reshape(1, -1)

            # Prediksi
            prediction = self.model.predict(input_array)[0]
            probabilities = self.model.predict_proba(input_array)[0]

            # Decode hasil prediksi
            personality_type = self.label_encoder.inverse_transform([prediction])[0]

            # Buat confidence scores untuk semua kelas
            classes = self.label_encoder.classes_
            confidence_scores = {
                classes[i]: f"{prob:.2%}" for i, prob in enumerate(probabilities)
            }

            # Format hasil yang lebih menarik
            max_prob = max(probabilities)

            # Emoji berdasarkan personality type
            personality_emoji = {"Extrovert": "🌟", "Introvert": "🤔", "Ambivert": "⚖️"}

            emoji = personality_emoji.get(personality_type, "🧠")

            # Format hasil dengan styling yang lebih menarik
            result = f"""
## {emoji} Hasil Prediksi Personality

### 🎯 **{personality_type.upper()}** 
**Confidence: {max_prob:.1%}**

---

### 📊 Detailed Confidence Scores:
"""

            # Tambahkan progress bar visual untuk setiap kelas
            for personality, score in confidence_scores.items():
                prob_value = probabilities[list(classes).index(personality)]
                bar_length = int(
                    prob_value * 20
                )  # Bar dengan panjang maksimal 20 karakter
                bar = "█" * bar_length + "░" * (20 - bar_length)

                # Tambahkan emoji untuk setiap personality
                class_emoji = personality_emoji.get(personality, "👤")
                result += f"\n**{class_emoji} {personality}:** {score}\n"
                result += f"`{bar}` {prob_value:.1%}\n"

            # Tambahkan interpretasi hasil
            result += "\n---\n\n### 💡 Interpretasi:\n"

            if max_prob >= 0.8:
                result += (
                    "🔥 **Sangat Yakin** - Model sangat confident dengan prediksi ini!"
                )
            elif max_prob >= 0.6:
                result += "✅ **Yakin** - Model cukup confident dengan prediksi ini."
            elif max_prob >= 0.4:
                result += (
                    "⚠️ **Cukup Yakin** - Ada beberapa kemungkinan personality type."
                )
            else:
                result += (
                    "❓ **Kurang Yakin** - Hasil prediksi tidak terlalu conclusive."
                )

            # Tambahkan tips berdasarkan personality
            result += f"\n\n### 🎭 Tentang {personality_type}:\n"

            if personality_type == "Extrovert":
                result += """
- 🗣️ Cenderung aktif dalam interaksi sosial
- ⚡ Mendapat energi dari berinteraksi dengan orang lain  
- 🎉 Senang menjadi pusat perhatian
- 👥 Memiliki lingkaran pertemanan yang luas
"""
            elif personality_type == "Introvert":
                result += """
- 🤫 Lebih suka aktivitas yang tenang dan privat
- 🔋 Membutuhkan waktu sendiri untuk mengisi energi
- 📚 Cenderung berpikir mendalam sebelum berbicara
- 👫 Memiliki lingkaran pertemanan yang kecil tapi dekat
"""
            else:
                result += """
- ⚖️ Memiliki karakteristik campuran extrovert dan introvert
- 🔄 Dapat beradaptasi dengan berbagai situasi sosial
- 🎯 Fleksibel dalam berinteraksi dengan orang lain
"""

            # Prepare data for bar plot
            plot_data = pd.DataFrame(
                {
                    "Personality": list(classes),
                    "Confidence": [
                        prob * 100 for prob in probabilities
                    ],  # Convert to percentage
                }
            )

            # Update status
            status_update = (
                f"✅ **Status:** Prediksi selesai - {personality_type} ({max_prob:.1%})"
            )

            return result, plot_data, True, status_update

        except Exception as e:
            error_msg = f"❌ Error dalam prediksi: {str(e)}"
            error_status = "❌ **Status:** Error dalam prediksi"
            # Return empty dataframe for error case
            empty_df = pd.DataFrame({"Personality": [], "Confidence": []})
            return error_msg, empty_df, False, error_status


def create_interface():
    """Membuat interface Gradio"""

    # Inisialisasi aplikasi
    app = PersonalityClassifierApp()

    # Cek apakah model berhasil dimuat
    if app.model is None:

        def error_fn(*args):
            return "❌ Model tidak dapat dimuat. Pastikan file model tersedia.", {}

        # Interface sederhana untuk error
        interface = gr.Interface(
            fn=error_fn,
            inputs=[gr.Number(label="Error", value=0)],
            outputs=[gr.Textbox(label="Status")],
            title="❌ Error Loading Model",
        )
        return interface

    # Interface utama
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Personality Classifier",
        css="""
        #result_output {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gradio-container {
            max-width: 1200px !important;
        }
        .gr-button-primary {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-weight: bold;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        .prediction-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        """,
    ) as interface:

        gr.Markdown(
            """
        # 🧠 Personality Classifier
        
        **Prediksi tipe personality berdasarkan karakteristik dan perilaku Anda!**
        
        Aplikasi ini menggunakan model Random Forest yang telah dilatih untuk memprediksi tipe personality 
        berdasarkan berbagai faktor seperti kebiasaan sosial, dan preferensi pribadi.
        """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Input Data Pribadi")

                time_alone = gr.Slider(
                    label="Waktu Sendirian (jam/hari)",
                    value=4,
                    minimum=0,
                    maximum=24,
                    step=0.5,
                    info="Berapa jam Anda menghabiskan waktu sendirian per hari?",
                )

                stage_fear = gr.Radio(
                    label="Takut Tampil di Depan Umum?",
                    choices=["Yes", "No"],
                    value="No",
                    info="Apakah Anda merasa takut atau gugup saat tampil di depan umum?",
                )

                social_events = gr.Slider(
                    label="Kehadiran Acara Sosial (0-10)",
                    value=5,
                    minimum=0,
                    maximum=10,
                    step=1,
                    info="Seberapa sering Anda menghadiri acara sosial? (0=tidak pernah, 10=sangat sering)",
                )

                going_outside = gr.Slider(
                    label="Frekuensi Keluar Rumah (0-10)",
                    value=5,
                    minimum=0,
                    maximum=10,
                    step=1,
                    info="Seberapa sering Anda keluar rumah? (0=tidak pernah, 10=sangat sering)",
                )

            with gr.Column():
                gr.Markdown("### 👥 Preferensi Sosial")

                drained_socializing = gr.Radio(
                    label="Merasa Lelah Setelah Bersosialisasi?",
                    choices=["Yes", "No"],
                    value="No",
                    info="Apakah Anda merasa lelah setelah berinteraksi sosial dalam waktu lama?",
                )

                friends_circle = gr.Number(
                    label="Ukuran Lingkaran Pertemanan",
                    value=10,
                    minimum=0,
                    maximum=50,
                    info="Berapa banyak teman dekat yang Anda miliki?",
                )

                post_frequency = gr.Slider(
                    label="Frekuensi Posting Media Sosial (1-10)",
                    value=5,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Seberapa sering Anda posting di media sosial? (1=jarang, 10=sangat sering)",
                )

                predict_btn = gr.Button(
                    "🔮 Prediksi Personality", variant="primary", size="lg"
                )

        with gr.Row():
            with gr.Column(scale=3):
                result_output = gr.Markdown(
                    label="📊 Hasil Prediksi",
                    value="""
                    
## 🎯 Personality Prediction Ready!

Masukkan data pribadi Anda pada form di sebelah kiri, kemudian klik tombol **🔮 Prediksi Personality** untuk melihat hasil analisis personality Anda.

### ✨ Fitur yang akan Anda dapatkan:
- 🎯 **Prediksi Akurat** dengan confidence score
- 📊 **Visualisasi** confidence untuk setiap tipe personality  
- 💡 **Interpretasi** hasil prediksi
- 🎭 **Penjelasan** karakteristik personality Anda
- 📈 **Progress bar** visual untuk setiap kemungkinan

**Siap untuk mengetahui personality Anda?** 🚀
                    """,
                    elem_id="result_output",
                )

            with gr.Column():
                gr.Markdown(
                    """
                ### 📚 Tentang Model
                
                **Model:** Random Forest Classifier  
                **Features:** 7 fitur input  
                **Akurasi:** Lihat file `Results/metrics.txt`  
                
                **Fitur yang digunakan:**
                - Waktu sendirian (jam/hari)
                - Takut tampil di depan umum
                - Kehadiran acara sosial (0-10)
                - Frekuensi keluar rumah (0-10)
                - Merasa lelah setelah bersosialisasi
                - Ukuran lingkaran pertemanan
                - Frekuensi posting media sosial (1-10)
                
                **Format Model:** Skops (.skops)  
                **Deployment:** Hugging Face Spaces
                """
                )

        # Event handler untuk prediksi dengan loading state
        def predict_with_loading(*inputs):
            # Update status ke loading
            yield "🔄 **Sedang memproses prediksi...**\n\nMohon tunggu sebentar..."

            # Jalankan prediksi
            result_text, plot_data, plot_visible, status_text = app.predict_personality(
                *inputs
            )

            # Return hasil prediksi
            yield result_text

        predict_btn.click(
            fn=predict_with_loading,
            inputs=[
                time_alone,
                stage_fear,
                social_events,
                going_outside,
                drained_socializing,
                friends_circle,
                post_frequency,
            ],
            outputs=[result_output],
        )

        gr.Markdown(
            """
        ---
        ### 🔗 Links
        - **GitHub Repository:** [Firmnm/Tugas-1-MLOps](https://github.com/Firmnm/Tugas-1-MLOps)
        - **Model Format:** Skops (scikit-learn compatible)
        - **Framework:** Gradio + Hugging Face Spaces
        
        *Dibuat untuk memenuhi Tugas 1 MLOps - Machine Learning Operations*
        """
        )

    return interface


if __name__ == "__main__":
    print("🚀 Launching Personality Classifier App...")

    # Get server configuration based on environment
    server_name, server_port = get_server_config()

    demo = create_interface()

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_api=False,
    )
