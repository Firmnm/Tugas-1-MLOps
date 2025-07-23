from fastapi import FastAPI, Request
import subprocess
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/trigger-retrain")
async def trigger_retrain(request: Request):
    try:
        # Safely parse body
        try:
            body = await request.json()
        except Exception:
            body = {"warning": "No JSON body or malformed JSON received"}

        print(" Alert diterima, mulai retrain:", body)

        # Jalankan train.py
        result = subprocess.run(
            ["python", "train.py", "--retrain", "--data_path", "Data/synthetic_ctgan_data.csv", "--old_data_path", "Data/personality_datasert.csv"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("Retraining berhasil.")
            print(result.stdout)
            return {"status": "success", "message": "Retraining completed", "output": result.stdout}
        else:
            print("Retraining gagal.")
            print(result.stderr)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Retraining failed", "error": result.stderr}
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "details": str(e)}
        )
