import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n--- Testing Health Check ---")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200

def test_analyze(claim_data, expected_status):
    response = requests.post(f"{BASE_URL}/api/claims/v1/analyze", json=claim_data)
    result = response.json()
    print(f"Status: {result['status']}, Type: {result.get('fraud_type', 'None')}, Reason: {result['reason_narrative']}")
    assert result['status'] == expected_status

def test_batch():
    print("\n--- Testing Batch Simulation ---")
    response = requests.post(f"{BASE_URL}/api/claims/v1/simulate-batch", json={"n": 10})
    result = response.json()
    print(f"Summary: {result['summary']}")
    assert response.status_code == 200
    assert result['total'] == 10

if __name__ == "__main__":
    test_health()

    print("\n--- Testing 5 SAFE Claims ---")
    safe_claims = [
        {"claim_id": f"SAFE-{i}", "doctor_notes": "Demam biasa", "admin_billing_code": ["R50.9"], "length_of_stay": 2, "total_charge": 1500000}
        for i in range(5)
    ]
    for claim in safe_claims:
        test_analyze(claim, "SAFE")

    print("\n--- Testing 5 FLAGGED Claims (Upcoding) ---")
    upcoding_claims = [
        {"claim_id": f"UPC-{i}", "doctor_notes": "Persalinan normal", "admin_billing_code": ["O14.9"], "length_of_stay": 3, "total_charge": 30000000}
        for i in range(5)
    ]
    for claim in upcoding_claims:
        test_analyze(claim, "FLAGGED")

    test_batch()
    print("\n[OK] All E2E Tests Passed!")
