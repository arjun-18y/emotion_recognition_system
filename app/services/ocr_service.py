def extract_text_from_image(file_obj):
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return None, "OCR dependencies missing. Install Pillow and pytesseract."

    try:
        image = Image.open(file_obj).convert("RGB")
        text = pytesseract.image_to_string(image)
        text = (text or "").strip()
        if not text:
            return None, "No readable text found in image."
        return text, None
    except Exception:
        return None, "Failed to read screenshot. Use a clear chat image (PNG/JPG)."
