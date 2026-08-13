import glob
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

app = FastAPI(title="WebGIS FORKESTRA Sultra API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 1. KONFIGURASI DIREKTORI
# ==============================================================================
BASE_DIR = "/Volumes/ssd/FORKESTRA"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def find_file_recursive(base_path, patterns):
  if not os.path.exists(base_path):
    return None
  for pattern in patterns:
    search_path = os.path.join(base_path, "**", pattern)
    matches = glob.glob(search_path, recursive=True)
    if matches:
      return sorted(matches)[0]
  return None


# ==============================================================================
# 2. BACA DATA CSV (PRIORITAS MASTER DASHBOARD)
# ==============================================================================
def load_master_csv():
  csv_master = find_file_recursive(
      BASE_DIR,
      [
          "*master*.csv",
          "*gabung*.csv",
          "*all*.csv",
          "*data_1_landcover_produksi.csv",
      ],
  )

  if csv_master:
    print(f"Memuat CSV Utama: {os.path.basename(csv_master)}")
    df = pd.read_csv(csv_master)
  else:
    print(
        "File master tidak ditemukan, mencoba mencari CSV 1 dan 2 terpisah..."
    )
    csv_1 = find_file_recursive(
        BASE_DIR, ["*data_1_landcover*.csv", "*data_1*.csv"]
    )
    csv_2 = find_file_recursive(BASE_DIR, ["*data_2_iklim*.csv", "*iklim*.csv"])

    df1 = pd.read_csv(csv_1) if csv_1 else pd.DataFrame()
    df2 = pd.read_csv(csv_2) if csv_2 else pd.DataFrame()

    if not df1.empty and not df2.empty:
      df1["id_kecamatan"] = df1["id_kecamatan"].astype(int)
      df2["id_kecamatan"] = df2["id_kecamatan"].astype(int)
      df1["tahun"] = df1["tahun"].astype(int)
      df2["tahun"] = df2["tahun"].astype(int)

      cols_to_use = [
          c
          for c in df2.columns
          if c not in df1.columns or c in ["id_kecamatan", "tahun"]
      ]
      df = pd.merge(
          df1, df2[cols_to_use], on=["id_kecamatan", "tahun"], how="left"
      )
    elif not df1.empty:
      df = df1
    else:
      print(
          "Error: Tidak ada file CSV yang ditemukan di /Volumes/ssd/FORKESTRA!"
      )
      return pd.DataFrame()

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
  return df


df_master = load_master_csv()


# ==============================================================================
# 3. HELPER AI INSIGHTS & DATA ENRICHMENT
# ==============================================================================
def generate_ai_insights(
    kecamatan: str, tahun: int, cur: dict, base: dict
) -> str:
  energi_tersedia = float(cur.get("energi_tersedia_mcal", 0) or 0)
  kebutuhan = float(cur.get("kebutuhan_energi_mcal", 0) or 0)
  neraca = float(cur.get("gap_energi_mcal", energi_tersedia - kebutuhan) or 0)
  sfai = float(cur.get("indeks_sfai", 0.5))

  pop_cur = float(cur.get("jumlah_penduduk_jiwa", 0) or 0)
  pop_base = float(base.get("jumlah_penduduk_jiwa", 0) or 0)
  delta_pop = pop_cur - pop_base
  pct_pop = ((delta_pop / pop_base) * 100) if pop_base > 0 else 0

  luas_sawah_cur = float(cur.get("luas_sawah_ha", 0) or 0)
  luas_sawah_base = float(base.get("luas_sawah_ha", 0) or 0)
  delta_sawah = luas_sawah_cur - luas_sawah_base

  yield_cur = float(cur.get("yield_pb_sawah_ton_ha", 0) or 0)
  yield_base = float(base.get("yield_pb_sawah_ton_ha", 0) or 0)

  status_pangan = (
      f"<b>SURPLUS</b> sebesar {neraca:,.0f} MCal"
      if neraca >= 0
      else f"<b>DEFISIT</b> sebesar {abs(neraca):,.0f} MCal"
  )

  if sfai <= 0.4:
    status_akses = "sangat rentan akibat kendala distribusi/aksesibilitas"
  elif sfai <= 0.6:
    status_akses = "cukup moderat namun perlu pengawasan rantai pasok"
  else:
    status_akses = "sangat baik dengan keterjangkauan tinggi"

  insight = (
      f"Pada tahun <b>{tahun}</b>, Kecamatan <b>{kecamatan}</b> diproyeksikan"
      f" berpopulasi <b>{pop_cur:,.0f} Jiwa</b> dengan neraca energi"
      f" {status_pangan}. Tingkat kerentanan logistik wilayah berada pada"
      f" kategori <b>{status_akses}</b> (Indeks SFAI: {sfai:.2f}).<br><br>"
  )
  insight += "<b>Rekomendasi AI & Intervensi Kebijakan:</b><ul>"

  if neraca < 0 and sfai <= 0.4:
    insight += (
        "<li><b>Prioritas Utama:</b> Segera perbaiki jaringan jalan tani dan"
        " logistik rantai pasok. Wilayah defisit ini harus disokong suplai dari"
        " kecamatan penyangga terdekat.</li>"
    )
  elif neraca < 0 and sfai > 0.4:
    insight += (
        "<li><b>Prioritas Intensifikasi:</b> Aksesibilitas wilayah sudah baik."
        " Terapkan benih adaptif perubahan iklim dan modernisasi irigasi untuk"
        " memacu yield beras lokal.</li>"
    )
  else:
    insight += (
        "<li><b>Prioritas Lumbung Surplus:</b> Pertahankan luasan LP2B dari"
        " alih fungsi dan optimalkan fasilitas gudang penyimpanan"
        " logistik.</li>"
    )

  if pct_pop > 10:
    insight += (
        f"<li><b>Tekanan Demografi:</b> Populasi bertambah"
        f" <b>+{pct_pop:.1f}% (+{delta_pop:,.0f} Jiwa)</b> dibanding baseline"
        " 2020.</li>"
    )

  if delta_sawah < 0:
    insight += (
        f"<li><b>Peringatan Konversi Lahan:</b> Lahan sawah menyusut"
        f" <b>{abs(delta_sawah):,.1f} Ha</b> dibanding 2020. Perlu penegakan"
        " tata ruang ketat.</li>"
    )

  if yield_base > 0 and yield_cur < yield_base:
    drop_pct = ((yield_base - yield_cur) / yield_base) * 100
    insight += (
        f"<li><b>Yield Gap Iklim:</b> Produktivitas bersih sawah diproyeksikan"
        f" turun {drop_pct:.1f}% ({yield_base:.2f} menjadi {yield_cur:.2f}"
        " Ton/Ha).</li>"
    )

  insight += "</ul>"
  return insight


def enrich_row_dict(row_series):
  d = row_series.fillna(0).to_dict()

  prod_sawah = float(d.get("total_pb_sawah_ton", 0))
  prod_kering = float(d.get("total_pb_kering_ton", 0))
  luas_sawah = float(d.get("luas_sawah_ha", 0))
  luas_kering = float(d.get("luas_kering_ha", 0))
  kebutuhan_mcal = float(d.get("kebutuhan_energi_mcal", 0))

  pop_val = float(
      d.get("jumlah_penduduk_jiwa", 0)
      or d.get("populasi", 0)
      or d.get("penduduk", 0)
      or 0
  )
  if pop_val == 0 and kebutuhan_mcal > 0:
    pop_val = kebutuhan_mcal / 459.9

  d["jumlah_penduduk_jiwa"] = round(pop_val, 2)
  d["delta_jumlah_penduduk_jiwa"] = round(
      float(d.get("delta_jumlah_penduduk_jiwa", 0)), 2
  )
  d["pct_jumlah_penduduk_jiwa"] = round(
      float(d.get("pct_jumlah_penduduk_jiwa", 0)), 2
  )

  d["luas_sawah"] = luas_sawah
  d["luas_kering"] = luas_kering
  d["produksi_bersih_sawah"] = prod_sawah
  d["produksi_bersih_kering"] = prod_kering

  d["yield_pk_sawah_ton_ha"] = round(
      float(d.get("yield_pk_sawah_ton_ha", 0)), 2
  )
  d["yield_pb_sawah_ton_ha"] = round(
      float(d.get("yield_pb_sawah_ton_ha", 0)), 2
  )
  d["delta_yield_pb_sawah_ton_ha"] = round(
      float(d.get("delta_yield_pb_sawah_ton_ha", 0)), 2
  )
  d["pct_yield_pb_sawah_ton_ha"] = round(
      float(d.get("pct_yield_pb_sawah_ton_ha", 0)), 2
  )

  d["yield_pk_kering_ton_ha"] = round(
      float(d.get("yield_pk_kering_ton_ha", 0)), 2
  )
  d["yield_pb_kering_ton_ha"] = round(
      float(d.get("yield_pb_kering_ton_ha", 0)), 2
  )
  d["delta_yield_pb_kering_ton_ha"] = round(
      float(d.get("delta_yield_pb_kering_ton_ha", 0)), 2
  )
  d["pct_yield_pb_kering_ton_ha"] = round(
      float(d.get("pct_yield_pb_kering_ton_ha", 0)), 2
  )

  d["energi_sawah"] = float(d.get("total_e_sawah_mcal", prod_sawah * 3570.0))
  d["energi_kering"] = float(d.get("total_e_kering_mcal", prod_kering * 2170.0))
  d["kebutuhan_energi"] = kebutuhan_mcal

  sfai_val = d.get("indeks_sfai")
  if sfai_val is None or pd.isna(sfai_val):
    sfai_val = d.get("indeks_k2_komposit", 0.5)

  d["indeks_sfai"] = float(sfai_val)
  d["indeks_k2_komposit"] = float(sfai_val)
  d["indeks_akses_informasi"] = float(d.get("indeks_akses_informasi", 0.5))

  return d


# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================
@app.get("/")
def read_root():
  html_file = find_file_recursive(BASE_DIR, ["index.html"])
  if html_file:
    response = FileResponse(html_file)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
  return {
      "status": "Online",
      "message": "Backend Active on SSD /Volumes/ssd/FORKESTRA",
  }


@app.get("/api/geojson")
def get_geojson():
  geojson_file = find_file_recursive(
      BASE_DIR, ["*.geojson", "kecamatan*.json", "shp*.json"]
  )
  if geojson_file and "package" not in geojson_file:
    with open(geojson_file, "r", encoding="utf-8") as f:
      return json.load(f)

  shp_file = find_file_recursive(
      BASE_DIR, ["*matched_raster.shp", "*kecamatan*.shp", "*.shp"]
  )
  if shp_file:
    import geopandas as gpd

    gdf = gpd.read_file(shp_file)
    return JSONResponse(content=json.loads(gdf.to_json()))

  return JSONResponse(
      status_code=404, content={"error": "File GeoJSON/SHP tidak ditemukan"}
  )


@app.get("/api/data/{kecamatan}/{tahun}")
def get_kecamatan_data(kecamatan: str, tahun: int):
  if df_master.empty:
    return JSONResponse(
        status_code=500, content={"error": "Data CSV tidak dimuat"}
    )

  kec_req = (
      kecamatan.lower().replace("kecamatan", "").replace("kec.", "").strip()
  )
  df_sub = df_master[
      (df_master["clean_kec"] == kec_req)
      & (df_master["tahun"] == int(tahun))
  ]
  df_2020 = df_master[
      (df_master["clean_kec"] == kec_req) & (df_master["tahun"] == 2020)
  ]

  if df_sub.empty:
    return {"status": "nodata"}

  cur_dict = enrich_row_dict(df_sub.iloc[0])
  base_dict = (
      enrich_row_dict(df_2020.iloc[0]) if not df_2020.empty else cur_dict
  )

  return {
      "status": "success",
      "current": cur_dict,
      "base_2020": base_dict,
      "ai_insights": generate_ai_insights(
          kecamatan, tahun, cur_dict, base_dict
      ),
  }


@app.get("/api/raster-bounds/{tahun}")
def get_raster_bounds(tahun: int):
  if int(tahun) == 2020:
    patterns = ["*landcover_2020*.tif", "*2020*reclass*.tif", "*2020*.tif"]
  else:
    patterns = [
        f"*LULC_Prediction_{tahun}*.tif",
        f"*LULC*{tahun}*.tif",
        f"*{tahun}*.tif",
    ]

  tif_path = find_file_recursive(BASE_DIR, patterns)

  if not tif_path:
    return JSONResponse(
        status_code=404,
        content={"error": f"Raster tahun {tahun} tidak ditemukan"},
    )

  with rasterio.open(tif_path) as src:
    bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    lng_min, lat_min, lng_max, lat_max = bounds_4326
    return [[lat_min, lng_min], [lat_max, lng_max]]


@app.get("/api/raster-png/{tahun}")
def get_raster_png(tahun: int):
  cache_file = os.path.join(CACHE_DIR, f"raster_cache_{tahun}.png")
  if os.path.exists(cache_file):
    return FileResponse(cache_file, media_type="image/png")

  if int(tahun) == 2020:
    patterns = ["*landcover_2020*.tif", "*2020*reclass*.tif", "*2020*.tif"]
  else:
    patterns = [
        f"*LULC_Prediction_{tahun}*.tif",
        f"*LULC*{tahun}*.tif",
        f"*{tahun}*.tif",
    ]

  tif_path = find_file_recursive(BASE_DIR, patterns)

  if not tif_path:
    return JSONResponse(
        status_code=404,
        content={"error": f"Raster tahun {tahun} tidak ditemukan"},
    )

  with rasterio.open(tif_path) as src:
    max_dim = max(src.width, src.height)
    scale = max_dim / 2000.0 if max_dim > 2000 else 1.0

    new_w = int(src.width / scale)
    new_h = int(src.height / scale)

    data = src.read(
        1,
        out_shape=(new_h, new_w),
        resampling=rasterio.enums.Resampling.nearest,
    )
    rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)

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
    for code, color in palette.items():
      mask = (data == code) | (data == float(code))
      rgba[mask] = color

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(cache_file, "PNG")
    return FileResponse(cache_file, media_type="image/png")