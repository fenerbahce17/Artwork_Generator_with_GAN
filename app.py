from flask import Flask, render_template, send_from_directory
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import cv2
import numpy as np
import os

# Flask uygulaması oluşturuluyor
app = Flask(__name__)

# Cihaz belirleme (GPU ya da CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_size = 100

# --- Generator Mimari --- 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_size, 64*8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64*8)
        self.conv2 = nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.conv3 = nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.conv4 = nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        return torch.tanh(self.conv5(x))

# --- Modeli Yükle --- 
generator = Generator().to(device)
checkpoint_path = "C:\\Users\\halil\\OneDrive\\Masaüstü\\DERİNOGRENME PROJE\\Sanat Eseri Projesi\\generator_model.chkpt\\160epochs.chkpt"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# --- Çözünürlük Artırma ve Keskinleştirme Fonksiyonu --- (OpenCV kullanarak)
import cv2
import numpy as np

def upscale_and_sharpen_image(input_path, output_path):
    # OpenCV ile resmi yükle
    image = cv2.imread(input_path)
    
    if image is None:
        raise ValueError("Resim yüklenemedi.")
    
    # Çözünürlük artırma (LANCZOS4 interpolasyonu kullanarak)
    height, width = image.shape[:2]
    new_width = width * 2  # Çözünürlüğü iki katına çıkar
    new_height = height * 2
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Keskinleştirme için Unsharp Mask filtresi (daha güçlü keskinleştirme)
    blurred_image = cv2.GaussianBlur(resized_image, (21, 21), 0)
    sharpened_image = cv2.addWeighted(resized_image, 1.5, blurred_image, -0.5, 0)

    # Sonuç resmi kaydet
    cv2.imwrite(output_path, sharpened_image)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    # Rastgele gürültü (latent space) oluştur
    noise = torch.randn(1, latent_size, 1, 1, device=device)
    
    # Modeli çalıştır
    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()
        save_image(fake_image, 'static/generated.png', normalize=True)

    # Çözünürlüğü artır ve keskinleştir
    upscale_and_sharpen_image('static/generated.png', 'static/upscaled_sharpened_generated.png')

    return render_template('index.html', image_generated=True)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
