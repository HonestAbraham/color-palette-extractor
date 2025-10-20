# color-palette-extractor

# from the project root (where palette_extractor.py lives)
python -m venv .venv
. .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python palette_extractor.py path/to/image.jpg --auto-k --k-min 4 --k-max 8 --css --json --posterize
