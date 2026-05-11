from pathlib import Path
import urllib.request

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

files = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest+.txt",
}

for filename, url in files.items():
    output_path = DATA_DIR / filename

    if output_path.exists():
        print(f"{filename} already exists. Skipping.")
        continue

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

print("Done.")