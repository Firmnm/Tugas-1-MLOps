import requests
import json

# Data yang akan diprediksi
data = [
    {
        "Time_spent_Alone": 4.0,
        "Stage_fear": 0,
        "Social_event_attendance": 4.0,
        "Going_outside": 6.0,
        "Drained_after_socializing": 0,
        "Friends_circle_size": 13.0,
        "Post_frequency": 5.0
    },
    {
        "Time_spent_Alone": 9.0,
        "Stage_fear": 1,
        "Social_event_attendance": 0.0,
        "Going_outside": 0.0,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 0.0,
        "Post_frequency": 3.0
    }
]

# Endpoint MLflow model
url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

# Format payload
payload = json.dumps({"dataframe_records": data})

# Kirim request POST ke endpoint
response = requests.post(url, headers=headers, data=payload)

# Cetak hasil prediksi
print("Predictions:", response.json())
