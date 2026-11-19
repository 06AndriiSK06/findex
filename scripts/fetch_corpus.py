import io
import urllib.request
import zipfile
from pathlib import Path

OUT_DIR = Path("data/python_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    ("3.12", "https://docs.python.org/3.12/archives/python-3.12-docs-text.zip"),
    ("3.13", "https://docs.python.org/3.13/archives/python-3.13-docs-text.zip"),
]

count = 0
for version, url in URLS:
    print(f"Downloading Python {version} docs...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        archive_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for member in zf.namelist():
            if not member.endswith(".txt"):
                continue
            rel_name = member.split("/", 1)[-1].replace("/", "_")
            target = OUT_DIR / f"py{version}_{rel_name}"
            target.write_bytes(zf.read(member))
            count += 1

print(f"Extracted {count} documents to {OUT_DIR}")
