#!/bin/bash
set -e

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Error: $1 tidak ditemukan. Pastikan sudah terinstall."
        exit 1
    fi
}

# Function to setup git config if not already set
setup_git_config() {
    if [ -z "$(git config --global user.email)" ]; then
        echo "⚙️ Setup git config..."
        git config --global user.email "autocommit@synthetic.local"
        git config --global user.name "synthetic-bot"
    fi
}

# Function to wait for synthetic data generation
wait_for_synthetic_data() {
    echo "⏳ Menunggu data sintetik pertama dibuat..."
    local max_wait=60  # maksimal 60 detik
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if [ -d "Data/versions" ] && [ "$(ls -A Data/versions 2>/dev/null)" ]; then
            echo "✅ Data sintetik pertama sudah dibuat."
            return 0
        fi
        sleep 5
        wait_time=$((wait_time + 5))
        echo "⏳ Masih menunggu... (${wait_time}s/${max_wait}s)"
    done
    
    echo "⚠️ Warning: Timeout menunggu data sintetik. Melanjutkan tanpa DVC versioning."
    return 1
}

echo "� [ENTRYPOINT] Memeriksa dependencies..."
check_command python
check_command dvc
check_command git

echo "�🚀 [ENTRYPOINT] Menjalankan generate_synthetic.py..."
python generate_synthetic.py &
SYNTHETIC_PID=$!

# Setup git config jika belum ada
setup_git_config

# Tunggu data sintetik pertama dibuat
if wait_for_synthetic_data; then
    echo "🧠 [ENTRYPOINT] Melakukan DVC versioning untuk versi terbaru..."
    
    # Pastikan git repository sudah diinisialisasi
    if [ ! -d ".git" ]; then
        echo "📁 Inisialisasi git repository..."
        git init
    fi
    
    # DVC versioning dengan error handling
    if dvc add Data/versions/; then
        git add Data/versions.dvc .gitignore
        if git commit -m "🔄 Auto-versioning synthetic data - $(date +%F_%T)"; then
            echo "✅ DVC versioning berhasil."
            dvc push || echo "⚠️ Gagal push ke remote (offline atau belum dikonfigurasi?)"
        else
            echo "⚠️ Gagal commit DVC changes (mungkin tidak ada perubahan)."
        fi
    else
        echo "⚠️ Gagal melakukan DVC add."
    fi
else
    echo "⚠️ Melanjutkan tanpa DVC versioning."
fi

echo "✅ [ENTRYPOINT] Menjalankan Gradio App..."
python App/app.py
