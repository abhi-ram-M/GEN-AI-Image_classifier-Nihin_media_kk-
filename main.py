import io, requests, torch, clip, streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz


SUPPORTED_CT = {"image/jpeg","image/png","image/webp","image/bmp","image/tiff"}
POPLER_BIN = None
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("model/clip_finetuned.pt", map_location=device)
model, preprocess = clip.load(ckpt["meta"]["model_name"], device=device)
for p in model.parameters(): p.requires_grad = False
model.eval()
classes = ckpt["meta"]["classes"]
feat_dim = model.visual.output_dim
classifier = torch.nn.Linear(feat_dim, len(classes)).to(device)
classifier.load_state_dict(ckpt["classifier_state"])
classifier.eval()

@torch.inference_mode()
def predict_batch_pils(pils, batch_size=32):
    out = []
    for i in range(0, len(pils), batch_size):
        batch = pils[i:i+batch_size]
        x = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        feats = model.encode_image(x).float()
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for p in probs:
            out.append((classes[p.argmax()], float(p.max())))
    return out

def extract_image_urls(url, max_imgs=50):
    soup = BeautifulSoup(requests.get(url, timeout=10).text, "html.parser")
    return [urljoin(url, img.get("src")) for img in soup.find_all("img") if img.get("src")][:max_imgs]

def fetch_pils_from_url(page_url, max_imgs=20):
    urls = extract_image_urls(page_url, max_imgs=max_imgs*3)
    imgs = []
    for u in urls:
        if len(imgs) >= max_imgs:
            break
        r = requests.get(u, timeout=10, stream=True)
        ct = r.headers.get("Content-Type","").split(";")[0].strip().lower()
        if ct in SUPPORTED_CT:
            imgs.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
    return imgs


def extract_pils_from_pdf(pdf_bytes, max_pages=10):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_by_page = []
    n = min(len(doc), max_pages)
    for i in range(n):
        page = doc.load_page(i)
        xrefs = page.get_images(full=True)
        for xref in xrefs:
            xref_id = xref[0]
            pix = fitz.Pixmap(doc, xref_id)
            if pix.alpha: 
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(pil_img)
            pix = None
    doc.close()
    return images

# UI
st.set_page_config(page_title="Medical vs Non-medical", layout="wide")
st.header("Medical vs Non-medical Image Classifier")

mode = st.radio("Input type", ["URL", "PDF"], horizontal=True)
images = []

if mode == "URL":
    url = st.text_input("Enter webpage URL")
    if st.button("Fetch images") and url:
        images = fetch_pils_from_url(url)
else:
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf is not None and st.button("Extract pages"):
        pdf_bytes = pdf.read() 
        images = extract_pils_from_pdf(pdf_bytes)

if images:
    preds = predict_batch_pils(images, batch_size=32)
    cols = st.columns(4)
    for i, (img, (label, conf)) in enumerate(zip(images, preds)):
        with cols[i % 4]:
            st.image(img, use_container_width=True)
            st.markdown(f"{label} ({conf*100:.1f}%)")
else:
    st.info("Provide a URL or PDF and click the button to classify images.")
