import json
import requests
from pathlib import Path
API_URL = "http://localhost:8000/analyze-crime"

# with open("C:\\Users\\singh\\Desktop\\CrimeScope-AI\\test_cases.json") as f:
#     test_cases = json.load(f)
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_FILE = BASE_DIR / "test_cases.json"

with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
    test_cases = json.load(f)
print("=" * 60)
print("CrimeScope AI — Test Suite")
print("=" * 60)

for tc in test_cases:
    print(f"\n[Test {tc['id']}] {tc['category']}")
    print(f"Input: {tc['input'][:80]}...")

    response = requests.post(API_URL, json={"description": tc["input"]})

    if response.status_code == 200:
        data = response.json()
        crimes = data["analysis"]["crime_type"]
        laws = [law["act"] for law in data["analysis"]["applicable_laws"]]
        time_taken = data["processing_time_seconds"]

        print(f" Status: SUCCESS | Time: {time_taken}s")
        print(f"   Crimes Found: {crimes}")
        print(f"   Laws Applied: {laws}")
    else:
        print(f" FAILED — Status: {response.status_code}")

print("\n" + "=" * 60)
print("Test Suite Complete!")