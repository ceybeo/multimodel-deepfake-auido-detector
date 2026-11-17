import io
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None   # çok büyük telefon fotoğrafları hata vermesin

MODEL_PATH = "efficientnet_b4_v3.pth"
ESIK_GERCEK = 0.30   # fake olasılığı bunun altındaysa "Gerçek"
ESIK_SAHTE = 0.70    # bunun üstündeyse "Yapay zekâ"

st.set_page_config(page_title="AI Görsel Dedektörü", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .sonuc { padding: 0.8rem 1rem; border-radius: 4px; font-size: 1.1rem;
             font-weight: 600; margin-top: 1rem; border-left: 4px solid; }
    .sahte { border-color: #DC2626; background: #FEF2F2; color: #991B1B; }
    .gercek { border-color: #16A34A; background: #F0FDF4; color: #166534; }
    .belirsiz { border-color: #D97706; background: #FFFBEB; color: #92400E; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def modeli_yukle():
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# Eğitimdeki doğrulama ve olcum3.py ile AYNI ön işleme
son = transforms.Compose([
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def hazirla(img):
    """Kısa kenar 600, JPEG kalite 90."""
    w, h = img.size
    s = 600 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fake_olasiligi(img):
    model = modeli_yukle()
    x = son(hazirla(img)).unsqueeze(0)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)[0]
    return float(p[0])   # sınıflar: 0 = Fake, 1 = Real


st.title("AI Üretimi Görsel Dedektörü")
st.write("Bir görsel yükleyin. Sistem görselin yapay zekâ ile üretilmiş olma olasılığını tahmin eder.")

yuklenen = st.file_uploader("Görsel seçin", type=["jpg", "jpeg", "png", "jfif"])

if yuklenen is not None:
    img = Image.open(yuklenen).convert("RGB")
    st.image(img, caption="Yüklenen görsel", use_container_width=True)

    if st.button("Analiz et"):
        with st.spinner("Analiz ediliyor..."):
            try:
                p = fake_olasiligi(img)
            except Exception as e:
                st.error(f"Analiz başarısız: {e}")
                st.stop()

        if p >= ESIK_SAHTE:
            sinif, metin = "sahte", "Yapay zekâ üretimi olma olasılığı yüksek"
        elif p <= ESIK_GERCEK:
            sinif, metin = "gercek", "Gerçek fotoğraf olma olasılığı yüksek"
        else:
            sinif, metin = "belirsiz", "Belirsiz: model karar veremedi"

        st.markdown(f'<div class="sonuc {sinif}">{metin}</div>', unsafe_allow_html=True)
        st.progress(p, text=f"Yapay zekâ olasılığı: %{100 * p:.1f}")

        st.caption(
            "Bu bir olasılık tahminidir, kesin hüküm değildir. Model, eğitimde görmediği "
            "yapay zekâ araçlarının çıktılarını kaçırabilir; aşırı keskin ya da yoğun işlenmiş "
            "gerçek fotoğraflara yanlışlıkla yüksek olasılık verebilir."
        )