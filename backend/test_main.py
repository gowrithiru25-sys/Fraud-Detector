"""
Unit tests for the fraud detection backend API.

Run with:  pytest test_main.py -v
(run this from inside backend/, with its venv active)
"""

from fastapi.testclient import TestClient
from app.main import app

# TestClient lets us send fake requests to our FastAPI app directly in
# Python -- no need to actually start uvicorn or use a real network call.
client = TestClient(app)

# A real legitimate transaction, taken from the test set (row 0).
LEGIT_EXAMPLE = {
    "Time": 0.0, "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.5363467396914,
    "V4": 1.3781552242443, "V5": -0.338320769942, "V6": 0.462387777762, "V7": 0.239598554061,
    "V8": 0.0986979012610, "V9": 0.363786969611, "V10": 0.0907941719789, "V11": -0.551599533260,
    "V12": -0.617800855762, "V13": -0.991389847235, "V14": -0.311169353699, "V15": 1.46817697209,
    "V16": -0.470400525259, "V17": 0.207971241929, "V18": 0.0257905801985, "V19": 0.403992960255,
    "V20": 0.251412098239, "V21": -0.018306777944, "V22": 0.277837575558, "V23": -0.110473910188,
    "V24": 0.0669280749146, "V25": 0.128539358273, "V26": -0.189114843888, "V27": 0.133558376740,
    "V28": -0.0210530534538, "Amount": 149.62,
}

# A real fraud transaction, taken from the test set.
# FRAUD_EXAMPLE = {
#     "Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201064158, "V3": -1.6098507322977,
#     "V4": 3.99790558754, "V5": -0.522187865064, "V6": -1.42654530849, "V7": -2.53738730624,
#     "V8": 1.39165724518, "V9": -2.77008927719, "V10": -2.77227214465, "V11": 3.20203320709,
#     "V12": -2.89990738849, "V13": -0.595221881324, "V14": -4.28925378244, "V15": 0.389724120274,
#     "V16": -1.14074717980, "V17": -2.83005567450, "V18": -0.0168224681808, "V19": 0.416955705862,
#     "V20": 0.126910559061, "V21": 0.517232370861, "V22": -0.0350493686052, "V23": -0.465211076182,
#     "V24": 0.320198198242, "V25": 0.0445191674731, "V26": 0.177839798284, "V27": 0.261145002567,
#     "V28": -0.143275874698, "Amount": 0.0,
# }
FRAUD_EXAMPLE = {'Time': 146022.0, 'V1': 0.908636658181293, 'V2': 2.84902401493181, 'V3': -5.64734296336341, 'V4': 6.00941477810058, 'V5': 0.216656395035464, 'V6': -2.39701442397293, 'V7': -1.81930788593969, 'V8': 0.338526987742346, 'V9': -2.81988277297887, 'V10': -4.06309810827338, 'V11': 2.94119009271712, 'V12': -6.15136219091534, 'V13': -1.98952853525001, 'V14': -9.15095100562097, 'V15': -0.604289998808266, 'V16': -1.95229039984463, 'V17': -2.89255533222526, 'V18': -0.912057960709434, 'V19': -1.56373996709776, 'V20': 0.241921294746754, 'V21': 0.407260461266497, 'V22': -0.397434852530952, 'V23': -0.0800058534116311, 'V24': -0.168596545191964, 'V25': 0.465058497798457, 'V26': 0.210509756241616, 'V27': 0.648704799002062, 'V28': 0.360224330288316, 'Amount': 1.18}

def test_root_endpoint_returns_ok():
    """The / endpoint should confirm the server is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud detection API is running"}


def test_predict_legitimate_transaction():
    """A known-legitimate transaction should be classified as legitimate."""
    response = client.post("/predict", json=LEGIT_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "legitimate"


def test_predict_fraud_transaction():
    """A known-fraud transaction should be classified as fraud."""
    response = client.post("/predict", json=FRAUD_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "fraud"


def test_predict_rejects_missing_field():
    """A request missing a required field (Amount) should fail validation,
    not crash the server or silently produce a wrong answer."""
    incomplete_transaction = LEGIT_EXAMPLE.copy()
    del incomplete_transaction["Amount"]

    response = client.post("/predict", json=incomplete_transaction)
    assert response.status_code == 422  # FastAPI's standard "validation failed" code


def test_predict_confidence_is_a_valid_probability():
    """Confidence should always be a number between 0 and 1, regardless
    of which class was predicted."""
    response = client.post("/predict", json=LEGIT_EXAMPLE)
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0
