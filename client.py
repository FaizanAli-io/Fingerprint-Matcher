import requests

BASE_URL = "http://localhost:8080"


def test_upload(filename):
    with open(f"data/{filename}.jpeg", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": f},
            data={"name": filename},
        )
        print("Upload response:", response.status_code, response.json())


def test_list():
    response = requests.get(f"{BASE_URL}/list")
    print("List response:", response.status_code, response.json())


def test_match(filename):
    with open(f"demo/{filename}.jpeg", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/match",
            files={"file": f},
        )
        print("Match response:", response.status_code, response.json())


def test_delete(id_to_delete):
    response = requests.delete(f"{BASE_URL}/delete/{id_to_delete}")
    print("Delete response:", response.status_code, response.json())


image_files = [
    "faizan-1",
    "faizan-2",
    "faizan-3",
    "mehdi-1",
    "mehdi-2",
    "mehdi-3",
]

if __name__ == "__main__":
    # [test_delete(i) for i in range(14, 17)]
    # for filename in image_files:
    #     test_upload(filename)
    # test_upload(filename)
    filename = "ali"
    test_match(filename)
    # test_list()
