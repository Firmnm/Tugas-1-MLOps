from App.app import PersonalityClassifierApp

def test_prediction_output():
    app = PersonalityClassifierApp()
    result, df, visible, status = app.predict_personality(
        4, "Yes", 5, 5, "No", 10, 5
    )
    assert "Hasil Prediksi" in result
    assert status.startswith("✅")
