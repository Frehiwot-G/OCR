import os
import cv2
import pytesseract
from PIL import Image
import difflib
import matplotlib.pyplot as plt
import csv
import re
from reportlab.lib.pagesizes import letter 
from reportlab.pdfgen import canvas


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Denoising
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive thresholding often works better than global Otsu for documents
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Optional: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def extract_text(image_array):
    """Extract text with optimized Tesseract parameters"""
    custom_config = r'--oem 3 --psm 6'  # OEM 3 = default engine, PSM 6 = assume uniform block of text
    return pytesseract.image_to_string(Image.fromarray(image_array), config=custom_config)

def calculate_accuracy(extracted, ground_truth):
    return difflib.SequenceMatcher(None, extracted.strip(), ground_truth.strip()).ratio()

def clean_text(text):
    """More comprehensive text cleaning"""
    text = text.lower()
    # Remove special characters but preserve basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\-\'"()]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def word_level_accuracy(extracted, ground_truth):
    extracted_words = clean_text(extracted).split()
    gt_words = clean_text(ground_truth).split()
    matches = sum(1 for w1, w2 in zip(extracted_words, gt_words) if w1 == w2)
    return matches / max(len(gt_words), 1)

def character_error_rate(extracted, ground_truth):
    """Fixed CER calculation with proper edit distance counting"""
    gt_clean = clean_text(ground_truth)
    ex_clean = clean_text(extracted)
    
    seq = difflib.SequenceMatcher(None, gt_clean, ex_clean)
    edits = 0
    
    for opcode in seq.get_opcodes():
        if opcode[0] != 'equal':
            if opcode[0] == 'replace':
                edits += max(opcode[2] - opcode[1], opcode[4] - opcode[3])
            elif opcode[0] == 'delete':
                edits += opcode[2] - opcode[1]
            elif opcode[0] == 'insert':
                edits += opcode[4] - opcode[3]
    
    return edits / max(len(gt_clean), 1)



def display_image_and_text(original_path, preprocessed_img, extracted_text, title="OCR Result"):
    """Displays original and preprocessed images with OCR text cleanly below."""
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1, 0.6]})
    fig.suptitle(title, fontsize=16)

    # Original image
    original = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    axs[0].imshow(original)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Preprocessed image
    axs[1].imshow(preprocessed_img, cmap="gray")
    axs[1].set_title("Preprocessed Image")
    axs[1].axis("off")

    # OCR Extracted Text (wrapped)
    axs[2].axis("off")
    axs[2].text(0, 1, f"OCR Extracted Text:\n{extracted_text.strip()}", fontsize=12,
                wrap=True, verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def load_image_ground_truth_pairs(image_folder, gt_folder):
    """More robust file loading with error handling"""
    image_files = []
    for f in os.listdir(image_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(f)
    image_files.sort()

    pairs = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        base_name = os.path.splitext(img_file)[0]
        gt_extensions = ['.txt', '.text', '.gt']
        gt_path = None
        
        for ext in gt_extensions:
            test_path = os.path.join(gt_folder, base_name + ext)
            if os.path.exists(test_path):
                gt_path = test_path
                break
        
        if not gt_path:
            print(f"Warning: Ground truth missing for {img_file}")
            continue

        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()
            pairs.append((img_file, img_path, ground_truth))
        except Exception as e:
            print(f"Error loading ground truth for {img_file}: {str(e)}")
    
    return pairs

def generate_pdf_report(results, output_path="ocr_report.pdf"):
    """Enhanced PDF reporting with better formatting"""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, y, "OCR Evaluation Report")
    y -= 40

    # Summary statistics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary Statistics:")
    y -= 20
    
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_word_acc = sum(r['word_accuracy'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Average Accuracy: {avg_accuracy:.2f}%")
    y -= 15
    c.drawString(50, y, f"Average Word Accuracy: {avg_word_acc:.2f}%")
    y -= 15
    c.drawString(50, y, f"Average Character Error Rate: {avg_cer:.2f}%")
    y -= 30

    # Detailed results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detailed Results:")
    y -= 20
    
    for result in results:
        if y < 100:  # New page if running out of space
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, f"Image: {result['image_name']}")
        y -= 15
        
        c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Accuracy: {result['accuracy']}% | Word Accuracy: {result['word_accuracy']}% | CER: {result['cer']}%")
        y -= 15
        
        # Truncate long texts for display
        gt_short = (result['ground_truth'][:150] + '...') if len(result['ground_truth']) > 150 else result['ground_truth']
        ocr_short = (result['ocr_text'][:150] + '...') if len(result['ocr_text']) > 150 else result['ocr_text']
        
        c.drawString(50, y, "Ground Truth:")
        y -= 15
        c.drawString(60, y, gt_short)
        y -= 15
        
        c.drawString(50, y, "OCR Output:")
        y -= 15
        c.drawString(60, y, ocr_short)
        y -= 25  # Extra space between entries

    c.save()
    print(f"\nPDF report saved to: {output_path}")

def run_real_image_ocr_evaluation(image_gt_pairs, output_csv="ocr_results.csv"):
    """Main evaluation function with better progress tracking"""
    results = []
    
    print("\n==== OCR Evaluation Results ====")
    print(f"Processing {len(image_gt_pairs)} images...\n")
    
    for idx, (image_name, image_path, ground_truth) in enumerate(image_gt_pairs, start=1):
        print(f"[{idx}] Processing: {image_name}")
        
        try:
            preprocessed = preprocess_image(image_path)
            extracted = extract_text(preprocessed)
            
            accuracy = calculate_accuracy(extracted, ground_truth) * 100
            word_acc = word_level_accuracy(extracted, ground_truth) * 100
            cer = character_error_rate(extracted, ground_truth) * 100
            
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Word Accuracy: {word_acc:.2f}%")
            print(f"  Character Error Rate: {cer:.2f}%")
            print("-" * 50)
            
            display_image_and_text(image_path, preprocessed, extracted, 
                                 title=f"{image_name} OCR Result")
            
            results.append({
                "image_name": image_name,
                "ocr_text": extracted.strip(),
                "ground_truth": ground_truth,
                "accuracy": round(accuracy, 2),
                "word_accuracy": round(word_acc, 2),
                "cer": round(cer, 2)
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            results.append({
                "image_name": image_name,
                "ocr_text": "ERROR",
                "ground_truth": ground_truth,
                "accuracy": 0.0,
                "word_accuracy": 0.0,
                "cer": 100.0
            })

    # Save results to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["image_name", "ocr_text", "ground_truth", "accuracy", "word_accuracy", "cer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")
    
    # Generate final PDF report
    generate_pdf_report(results)

def main():
    """Main function with directory validation"""
    image_folder = "images"
    gt_folder = "ground_truths"
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' not found!")
        return
        
    if not os.path.exists(gt_folder):
        print(f"Error: Ground truth folder '{gt_folder}' not found!")
        return
    
    image_gt_pairs = load_image_ground_truth_pairs(image_folder, gt_folder)
    
    if not image_gt_pairs:
        print("No valid image/ground truth pairs found!")
        return
    
    run_real_image_ocr_evaluation(image_gt_pairs)

if __name__ == "__main__":
    main()