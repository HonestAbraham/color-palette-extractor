# 🎨 Color Palette Extractor

**Color Palette Extractor** is a command-line tool that analyzes any image and extracts its **dominant colors**, generating a visual color palette along with optional **CSS**, **JSON**, and **posterized** outputs.  

It’s perfect for designers, developers, and creators who want to quickly build color themes from photos, artwork, or product images.

---

## 🚀 Features

- 🖼️ Extract the most dominant colors from any image  
- ⚙️ Automatically determine the best number of clusters (`--auto-k`)  
- 🎨 Save palette images with HEX codes and percentage share  
- 💻 Export CSS color variables and JSON color data  
- 🧩 Generate a “posterized” version of the image recolored by the palette  
- 🧼 Ignore near-white or near-black pixels to remove background noise  
- ⚡ Fast — uses MiniBatch K-Means clustering via scikit-learn  

---

## 🧩 Tech Stack

| Component | Purpose |
|------------|----------|
| **Python 3.11** | Core language |
| **OpenCV (`cv2`)** | Image I/O, resizing, and color conversions |
| **NumPy** | Array and math operations |
| **scikit-learn** | K-Means clustering & silhouette scoring |
| **Pillow (PIL)** | Palette image generation and posterizing |
| **Matplotlib** *(optional)* | Visual previews (`--show` flag) |

---

## ⚙️ Installation

From the project root (where `palette_extractor.py` lives):

```bash
# Create and activate a virtual environment
python -m venv .venv
. .venv/Scripts/activate   # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
