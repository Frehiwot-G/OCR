# 🧠 Optical Character Recognition (OCR) with Tesseract

This project performs Optical Character Recognition (OCR) using **Tesseract OCR** along with **OpenCV** preprocessing techniques. It extracts text from printed images, compares it to ground truth, calculates accuracy metrics, and exports results as CSV and PDF.

---

## 📁 Project Structure


│
├── images/ # Folder containing input images
├── ground_truths/ # Folder for Ground truth for each image
├── requirements.txt # Python package dependencies
├── OCR.py # Main script to run OCR + evaluation
├── ocr_results.csv # Output: CSV with text, ground truth, accuracy
├── ocr_report.pdf # Output: PDF visual summary
├── README.md # This file


---

## 📦 Setup Instructions

### 1. ✅ Install Tesseract OCR

- **Windows**: Download from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract), then update this line in `OCR.py`:

  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

**Linux/macOS:**
sudo apt install tesseract-ocr       # Debian/Ubuntu
brew install tesseract               # macOS


## 📦 Install Python Dependencies

pip install -r requirements.txt


## Run the OCR script

python OCR.py