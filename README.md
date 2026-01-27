# PDF Translator - Azure OpenAI Streamlit App

A standalone Streamlit application for translating PDF documents while preserving **ALL** formatting including fonts, sizes, colors, styles, images, tables, layouts, headers, footers, spacing, alignment, and positioning.

## Features

- 🌐 **Multi-language Support**: Translate between multiple languages
- 🎨 **Format Preservation**: Maintains pixel-perfect formatting of original documents
- 📊 **Layout Preservation**: Preserves formulas, charts, tables, headers, footers, and annotations
- 🖼️ **Image Preservation**: Keeps all images and graphics unchanged
- 📐 **Precise Positioning**: Translated text placed in exact original positions
- ⚡ **Progress Tracking**: Real-time progress indicators for multi-page PDFs
- 🔒 **Secure**: Uses Azure OpenAI with environment variable configuration

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Azure OpenAI account with:
  - Azure OpenAI endpoint
  - API key
  - Deployment name (e.g., gpt-4o-mini)
  - API version

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PDFMathTranslate
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env .env
   ```
   
   Edit `.env` and add your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
   AZURE_OPENAI_API_VERSION=2024-06-01
   ```

## Usage

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   The app will automatically open at `http://localhost:8500`

3. **Select languages**:
   - Choose source language (e.g., English)
   - Choose target language (e.g., Chinese)

4. **Upload and translate**:
   - Click "Upload PDF File" and select your PDF
   - Click "Translate"
   - Wait for translation to complete
   - Download the translated PDF (monolingual or bilingual)

## How It Works

1. **PDF Parsing**: Extracts text with precise coordinates and formatting information
2. **Layout Detection**: Uses ONNX model to detect document structure (text, formulas, images, tables)
3. **Text Extraction**: Identifies text blocks while preserving context and formatting
4. **Translation**: Uses Azure OpenAI to translate text while maintaining context
5. **PDF Reconstruction**: Places translated text in exact original positions
6. **Format Preservation**: Maintains all non-text elements (images, tables, layouts) unchanged

## Output Files

- **Monolingual PDF** (`filename-mono.pdf`): Translated PDF with original formatting
- **Bilingual PDF** (`filename-dual.pdf`): Bilingual PDF with original and translated text side-by-side

## Configuration Options

### Translation Threads
Adjust the number of parallel translation threads (1-8) in the sidebar. More threads = faster translation but higher API usage.

### Font Subsetting
- **Enabled** (default): Reduces file size by including only used characters
- **Disabled**: Includes full fonts (larger files but better compatibility)

## Troubleshooting

### Model Download Issues
If you encounter network issues downloading the ONNX model, set:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Azure OpenAI Errors
- Verify your endpoint URL is correct
- Check that your API key is valid
- Ensure your deployment name matches your Azure resource
- Verify API version compatibility

### Memory Issues
For large PDFs, you may need to:
- Reduce translation threads
- Process PDFs page by page
- Increase system memory

## Technical Details

- **PDF Processing**: Uses pdfminer.six for parsing and PyMuPDF for manipulation
- **Layout Detection**: DocLayout-YOLO ONNX model for structure detection
- **Translation**: Azure OpenAI API with context-aware prompts
- **Font Handling**: Automatic font downloading for target languages

## License

AGPL-3.0

## Acknowledgments

- Document parsing: [Pdfminer.six](https://github.com/pdfminer/pdfminer.six)
- Document manipulation: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- Layout parsing: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- Multilingual Fonts: [Go Noto Universal](https://github.com/satbyy/go-noto-universal)
