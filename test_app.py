import pytest
from fastapi.testclient import TestClient
from app import app, load_model
import joblib


@pytest.fixture
def client():
    return TestClient(app)


def test_load_model_success():
 # Проверяем, что функция load_model возвращает загруженную модель
    loaded_model = joblib.load('model.joblib')
    assert loaded_model is not None


def test_root_get(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200


def test_prediction_success(client: TestClient):
    payload = [{
        'credit_score': 585,
        'geography': 'France',
        'gender': 'Male',
        'age': 36,
        'tenure': 7.0,
        'balance': 0.0,
        'num_of_products': 2,
        'has_cr_card': 1,
        'is_active_member': 0,
        'estimated_salary': 94283.09
    }]
    response = client.post("/prediction", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
    answer = response.json()["answer"]
    # Проверяем, что в ответе возвращается список из 0 и/или 1:
    if isinstance(answer, list):
        for item in answer:
            assert item in [0, 1]
    else:
        raise AssertionError("Unexpected answer format")


def test_prediction_invalid_data(client: TestClient):
    payload = [{
        "credit_score": "abc",  # Ошибка: должно быть int
        'geography': 'France',
        'gender': 'Male',
        'age': 36,
        'tenure': 7.0,
        'balance': 0.0,
        'num_of_products': 2,
        'has_cr_card': 1,
        'is_active_member': 0,
        'estimated_salary': 94283.09
    }]
    response = client.post("/prediction", json=payload)
    assert response.status_code == 422  # Ошибка валидации
