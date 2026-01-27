# Quick Start Guide

## Prerequisites

1. Python 3.10, 3.11, or 3.12 installed
2. Azure OpenAI account with:
   - Endpoint URL
   - API Key
   - Deployment name (e.g., `gpt-4o-mini`)
   - API version (e.g., `2024-06-01`)

## Installation Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Azure OpenAI credentials**:
   
   Option A: Using `.env` file (recommended):
   ```bash
   cp .env .env
   # Edit .env and add your credentials
   ```

   Option B: Enter credentials in the Streamlit UI sidebar

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**:
   The app will automatically open at `http://localhost:8501`

## Usage

1. **Configure Azure OpenAI** (if not using .env):
   - Enter Azure endpoint in sidebar
   - Enter API key
   - Specify deployment name
   - Set API version

2. **Select languages**:
   - Source language (e.g., English)
   - Target language (e.g., Chinese)

3. **Upload PDF**:
   - Click "Upload PDF File"
   - Select your PDF document

4. **Translate**:
   - Click "🚀 Translate PDF"
   - Wait for translation to complete
   - Download translated PDF (monolingual or bilingual)

## Troubleshooting

### Model Download Issues
If you encounter network issues downloading the ONNX model:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Azure OpenAI Errors
- Verify endpoint URL format: `https://your-resource.openai.azure.com`
- Check API key is valid and has proper permissions
- Ensure deployment name matches your Azure resource
- Verify API version is supported

### Import Errors
If you see import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Output Files

- **Monolingual PDF** (`filename-mono.pdf`): Translated PDF with original formatting preserved
- **Bilingual PDF** (`filename-dual.pdf`): Bilingual PDF with original and translated text

## Features

✅ Preserves ALL PDF formatting:
- Fonts, sizes, colors, styles
- Images and graphics
- Tables and layouts
- Headers and footers
- Spacing and alignment
- Precise positioning
