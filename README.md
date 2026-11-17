# Restaurant Intelligence

A Streamlit application for restaurant analytics, rating prediction, and restaurant discovery.

## Modules
- Restaurant analytics for an uploaded CSV
- Rating prediction: train a baseline regression model from the uploaded dataset
- Restaurant finder: search uploaded restaurant records by cost-for-two range and minimum rating
- Food image recognition: uses a general pretrained Vision Transformer via Hugging Face Transformers
- Menu OCR: uses EasyOCR to extract text from uploaded menu images
- Visual image analysis: brightness, contrast, sharpness indicator, and simple colour statistics
- Model evaluation: reference metrics from the original presentation plus metrics from a newly trained model

The sidebar groups modules by task and includes a shared CSV uploader and loaded-dataset summary. Upload a dataset once, then navigate between analytics and prediction modules without uploading it again. The restaurant finder returns up to six matching records ordered by rating, then cost; it does not use votes or invent restaurants.

## Run locally
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

### Uploading from a phone to a Windows PC

The upload control uses the browser's normal file picker, so a PC can browse its files and a phone can choose from its Photos or Files app. To open an app running on your PC from a phone, connect both devices to the same trusted Wi-Fi network and start Streamlit with:

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Find the PC's IPv4 address with `ipconfig`, then open `http://<PC-IPv4-address>:8501` in the phone's browser. If Windows Firewall asks, allow access only on a private network. This local setup has no sign-in, so do not expose it to public networks or upload sensitive data.

## Dataset
Upload your Zomato CSV from the sidebar or the Restaurant Analytics module. The app attempts to identify common column names such as `Aggregate rating`, `Votes`, `Cuisines`, `City`, `Restaurant Name`, and `Average Cost for two`.

## Important limitations
- Food recognition downloads a pretrained Food-101 model on first use and requires internet access. Its label set is limited and it may not recognise every dish.
- EasyOCR may download model weights on first use; text and prices should be checked manually.
- The visual module calculates simple image statistics. It does not infer taste, hygiene, freshness, or actual food quality.
- Restaurant discovery requires cost-for-two and rating columns; city, restaurant name, and cuisine are shown when available. Cost is displayed in rupees for Zomato-style INR datasets.
- Rating-model selection uses 5-fold cross-validation on the training split and reports a separate holdout score. Metrics depend on the uploaded data and should not be treated as guarantees.
- For a serious deployment, improve data validation, add model persistence, test models with cross-validation, and review possible data leakage.
