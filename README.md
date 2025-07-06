# PII Masker - Advanced ID Document Image Processing

A web-based application that automatically detects and masks personal information (PII) from scanned ID document images using advanced OCR and regex techniques.

## Features

- **Advanced Image Preprocessing**: Multi-threshold processing, denoising, deskewing, and smart resizing
- **Enhanced OCR Pipeline**: Multiple PSM modes, confidence filtering, and overlapping text merging
- **Advanced PII Detection**: 
  - **SSN**: Multiple formats with validation (excludes 000, 666, 900+)
  - **Credit Cards**: Luhn algorithm validation for major card types
  - **Phone Numbers**: US and international formats
  - **Email**: RFC-compliant pattern matching
  - **Names**: Context-aware detection with spaCy NER
  - **Addresses**: Street address pattern recognition
  - **DOB**: Multiple date format support
- **Multiple Masking Options**: Black rectangles, Gaussian blur, or pixelation
- **MongoDB Storage**: Save processing history and detected PII
- **Modern Web Interface**: Professional UI with real-time feedback

## Prerequisites

### Windows Installation

1. **Python 3.8+** (already installed)
2. **Tesseract OCR** (already installed via winget)
3. **MongoDB** (optional - for storing processing history)

### Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Setup

1. **Tesseract Configuration**: The application is configured to use Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`

2. **MongoDB** (Optional):
   - Install MongoDB locally or use MongoDB Atlas
   - Update `MONGO_URL` in `models.py` if using a different connection

3. **Directory Structure**: The application creates:
   - `static/uploads/` - Original images
   - `static/masked/` - Processed images with PII masked

## Usage

### Start the Application

```bash
uvicorn main:app --reload
```

### Access the Web Interface

Open your browser and go to: http://localhost:8000

### Upload and Process Images

1. Click "Choose File" and select an ID document image
2. Select your preferred masking method (Black, Blur, or Pixelate)
3. Click "Upload & Process"
4. View the original and masked images side-by-side
5. Download the masked image
6. Review detected PII information with confidence scores

## API Endpoints

- `GET /` - Main upload interface
- `POST /upload` - Process uploaded image and return masked version

## Technical Details

### Advanced Image Processing Pipeline

1. **Multi-threshold Preprocessing** (`preprocess_image`):
   - Smart resizing for small images
   - Advanced denoising (FastNlMeansDenoising, Gaussian, Median)
   - Multiple thresholding methods (Gaussian, Mean, Otsu)
   - Enhanced deskew correction

2. **Enhanced OCR** (`extract_ocr_data`):
   - Multiple PSM modes (6, 8, 13) for different text layouts
   - Confidence-based filtering (>30% threshold)
   - Overlapping text merging
   - Optimized Tesseract configuration

3. **Advanced PII Detection** (`detect_pii`):
   - **SSN Patterns**: `\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b`
   - **Credit Card**: Luhn algorithm validation for major card types
   - **Phone Numbers**: `\b\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b`
   - **Email**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
   - **Names**: Context-aware detection with spaCy NER
   - **Addresses**: Street address pattern recognition
   - **DOB**: Multiple date format support

4. **Multiple Masking Options** (`mask_pii`):
   - **Black Rectangles**: Complete coverage with padding
   - **Gaussian Blur**: Strong blur for unreadable text
   - **Pixelation**: Block-based pixelation effect

### Advanced Regex Patterns

#### SSN Detection
```regex
# Standard format with validation
\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b

# Space separated
\b(?!000|666|9\d{2})\d{3}\s(?!00)\d{2}\s(?!0000)\d{4}\b

# Dot separated
\b(?!000|666|9\d{2})\d{3}\.(?!00)\d{2}\.(?!0000)\d{4}\b

# No separators (with validation)
\b(?!000|666|9\d{2})\d{9}\b
```

#### Credit Card Detection
```regex
# Major card types with Luhn validation
\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b

# Generic 16-digit format
\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b
```

#### Phone Number Detection
```regex
# US format
\b\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b

# International format
\b\+?1[-. ]?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b
```

### File Structure

```
/
├── main.py              # FastAPI application with masking options
├── ocr_utils.py         # Advanced OCR and PII detection logic
├── models.py            # MongoDB integration
├── requirements.txt     # Python dependencies
├── test_ocr.py          # Comprehensive test suite
├── templates/
│   └── index.html       # Enhanced web interface
└── static/
    ├── uploads/         # Original images
    └── masked/          # Processed images
```

## Advanced Features

### Context-Aware Detection
- **Name Detection**: Uses spaCy NER + context analysis
- **Proximity Analysis**: Checks for nearby keywords
- **Overlapping PII Merging**: Combines adjacent PII elements

### Validation Algorithms
- **Luhn Algorithm**: Credit card number validation
- **SSN Validation**: Excludes invalid patterns (000, 666, 900+)
- **Name Validation**: Sophisticated name pattern recognition

### Performance Optimizations
- **Multi-threshold Processing**: Tests multiple preprocessing methods
- **Confidence Filtering**: Only processes high-confidence text
- **Smart Text Merging**: Combines overlapping OCR results

## Troubleshooting

### Tesseract Issues

If you get "TesseractNotFoundError":
1. Verify Tesseract is installed: `Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"`
2. Check the path in `ocr_utils.py` matches your installation
3. Restart the application after path changes

### MongoDB Issues

If MongoDB is not available:
- The application will still work for image processing
- Only the history storage will be affected
- Check MongoDB connection in `models.py`

### Image Processing Issues

- Ensure images are clear and well-lit
- Text should be readable by human eyes
- Supported formats: JPG, PNG, BMP, TIFF
- Minimum recommended resolution: 800px width

### PII Detection Issues

- **Low Detection Rate**: Try different masking methods
- **False Positives**: Check confidence scores in results
- **Missing PII**: Ensure text is clearly visible in image

## Performance Statistics

- **OCR Accuracy**: 60-85% confidence range
- **PII Detection**: Supports 7+ PII types
- **Processing Speed**: ~2-5 seconds per image
- **Masking Quality**: Professional-grade results

## Future Enhancements

- **Machine Learning Models**: Custom-trained OCR models
- **Batch Processing**: Multiple image processing
- **API Rate Limiting**: Production-ready security
- **User Authentication**: Multi-user support
- **Custom PII Patterns**: User-defined detection rules
- **Advanced Masking**: Custom patterns and colors
- **Export Options**: PDF, ZIP archives
- **Audit Logging**: Detailed processing history

## License

This project uses open-source tools and libraries. See individual component licenses for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test output
3. Verify your image quality
4. Check Tesseract installation 