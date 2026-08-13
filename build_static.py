import glob
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_RASTER_DIR = os.path.join(BASE_DIR, "static_rasters")
os.makedirs(OUTPUT_RASTER_DIR, exist_ok=True)


def find_file(patterns):
    for p in patterns:
        matches = glob.glob(os.path.join(BASE_DIR, "**", p), recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


# ==========================================
# 1. Konversi Data CSV ke JSON Statis
# ==========================================
print("▶ [1/2] Memproses file CSV tabular...")
csv_path = find_file(["*master*.csv", "*data_1_landcover_produksi.csv", "*.csv"])

if not csv_path:
    print("❌ Error: File CSV tidak ditemukan!")
else:
    print(f"   Menggunakan CSV: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "nama_kecamatan" in df.columns:
        df["clean_kec"] = (
            df["nama_kecamatan"]
            .astype(str)
            .str.lower()
            .str.replace("kecamatan", "", regex=False)
            .str.replace("kec.", "", regex=False)
            .str.strip()
        )
    json_path = os.path.join(BASE_DIR, "data_web.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"   ✔ Berhasil membuat: {os.path.basename(json_path)}")

# ==========================================
# 2. Konversi TIFF ke PNG (Resolusi 100% Murni)
# ==========================================
print("\n▶ [2/2] Mengonversi Raster TIFF ke PNG (100% Resolusi Asli)...")
years = [2020, 2025, 2030, 2035, 2040, 2045]
bounds_dict = {}

palette = {
    1: [46, 125, 50, 220],
    2: [143, 168, 118, 190],
    3: [224, 211, 193, 190],
    4: [255, 102, 0, 255],
    5: [255, 255, 0, 255],
    6: [170, 0, 255, 220],
    7: [255, 0, 0, 255],
    8: [2, 136, 209, 220],
}

for th in years:
    pat = (
        ["*2020*reclass*.tif", "*2020*.tif"]
        if th == 2020
        else [f"*Prediction_{th}*.tif", f"*{th}*.tif"]
    )
    tif = find_file(pat)
    if tif:
        print(f"   -> Memproses Raster Tahun {th} ({os.path.basename(tif)})...")
        with rasterio.open(tif) as src:
            if not bounds_dict:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                bounds_dict["bounds"] = [[b[1], b[0]], [b[3], b[2]]]

            # Membaca data 100% ukuran asli tanpa downsampling
            data = src.read(1)
            rgba = np.zeros((src.height, src.width, 4), dtype=np.uint8)

            for code, col in palette.items():
                mask = (data == code) | (data == float(code))
                rgba[mask] = col

            img = Image.fromarray(rgba, mode="RGBA")
            img.save(
                os.path.join(OUTPUT_RASTER_DIR, f"raster_{th}.png"),
                "PNG",
                optimize=True,
            )

bounds_file = os.path.join(BASE_DIR, "raster_bounds.json")
with open(bounds_file, "w") as f:
    json.dump(bounds_dict, f, indent=2)

print(f"   ✔ Koordinat batas peta tersimpan di: {os.path.basename(bounds_file)}")
print("\n✅ SELESAI! Seluruh aset statis berhasil diekstrak.")