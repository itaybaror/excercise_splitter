import os
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Step 1: Convert PDF to Images using PyMuPDF
def pdf_to_images(pdf_path, dpi=300):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Step 2: Detect List Items Using OCR
def detect_list_items(image, min_conf=70, list_markers=None):
    if list_markers is None:
        list_markers = ['1.', '2.', '3.', '4.', '5.', '6', '7', '8', 'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.']

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Run OCR to detect text with bounding boxes
    custom_oem_psm_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_oem_psm_config, output_type=pytesseract.Output.DICT)

    items = []
    current_item = ""  # Initialize as empty string instead of None
    item_boxes = []
    current_item_marker = None
    page_height = image.size[1]  # Height of the page in pixels

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > min_conf and data['text'][i].strip():
            text = data['text'][i].strip()
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Check if the current text starts a new item based on list marker
            if any(text.startswith(marker) for marker in list_markers):  # Detecting a list marker
                if current_item:  # Save the previous item if it's complete
                    items.append((current_item_marker, current_item, item_boxes))

                # Start a new item
                current_item_marker = text
                current_item = text
                item_boxes = [(x, y, x + w, y + h)]
            else:
                # Add the current line to the current item
                current_item += " " + text
                item_boxes.append((x, y, x + w, y + h))

    # Add the last item
    if current_item:
        items.append((current_item_marker, current_item, item_boxes))

    return items

# Step 3: Crop and Save Each Item as an Image
def crop_and_save_items(image, items, output_folder, page_num):
    os.makedirs(output_folder, exist_ok=True)
    for i, (marker, item, boxes) in enumerate(items, start=1):
        # Combine all the bounding boxes into a single rectangle
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[2] for box in boxes)
        max_y = max(box[3] for box in boxes)

        # Crop the image based on the bounding box
        cropped = image.crop((min_x, min_y, max_x, max_y))

        # Save the cropped image as a separate file
        output_path = os.path.join(output_folder, f"page_{page_num}_item_{i}.png")
        cropped.save(output_path)
        print(f"Saved: {output_path}")

# Step 4: Full Workflow to Process PDF
def process_pdf_to_images(pdf_path, output_folder, dpi=300, min_conf=70):
    images = pdf_to_images(pdf_path, dpi)

    for page_num, image in enumerate(images, start=1):
        print(f"Processing page {page_num}...")
        items = detect_list_items(image, min_conf=min_conf)
        crop_and_save_items(image, items, output_folder, page_num)

# Input PDF and Output Folder
pdf_path = "ex3.pdf"
output_folder = "extracted_items"

# Run the process
process_pdf_to_images(pdf_path, output_folder)
