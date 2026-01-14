import argparse
import os
import sys
from typing import Tuple, Literal, Callable, Optional

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def to_float_series(s: pd.Series) -> pd.Series:
    # Normalize decimal separators and coerce to float
    s = s.astype(str).str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def read_sheets(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df_k = pd.read_excel(path, sheet_name='KACC', header=0)
        df_p = pd.read_excel(path, sheet_name='Pos', header=0)
    except Exception as e:
        log(f"Hata: Excel okuma başarısız: {e}")
        raise
    return df_k, df_p


def extract_coords(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Expect latitude in column J (index 9) and longitude in column K (index 10)
    if df.shape[1] < 11:
        raise ValueError(f"{label} sayfasında en az 11 sütun bekleniyor (J ve K koordinatlar).")

    lat = to_float_series(df.iloc[:, 9])
    lon = to_float_series(df.iloc[:, 10])

    # Basic identity columns for output: first two columns assumed as Musteri No and Musteri Adı
    id_cols = df.iloc[:, :2].copy()
    id_cols.columns = ['Musteri No', 'Musteri Adı']

    mask = lat.notna() & lon.notna()
    clean_df = df.loc[mask].reset_index(drop=True)
    id_cols = id_cols.loc[mask].reset_index(drop=True)
    lat = lat.loc[mask].to_numpy(dtype=float)
    lon = lon.loc[mask].to_numpy(dtype=float)

    if len(lat) == 0:
        raise ValueError(f"{label} sayfasında geçerli koordinat bulunan satır yok.")

    return id_cols, lat, lon


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    # Returns distance matrix in meters of shape (len(lat1), len(lat2))
    R = 6371000.0
    to_rad = np.pi / 180.0

    lat1r = lat1[:, None] * to_rad
    lon1r = lon1[:, None] * to_rad
    lat2r = lat2[None, :] * to_rad
    lon2r = lon2[None, :] * to_rad

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def compute_nearest(
    df_kacc: pd.DataFrame,
    df_pos: pd.DataFrame,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")
        logger("Mesafe matrisi hesaplanıyor")
    dist = haversine_matrix(k_lat, k_lon, p_lat, p_lon)
    if logger:
        logger("En yakın indeksler bulunuyor")
    argmin = np.argmin(dist, axis=1)  # index of nearest POS for each KACC
    min_dist = dist[np.arange(dist.shape[0]), argmin]

    # Build result dataframe
    res = pd.DataFrame({
        'KACC Musteri No': kacc_ids['Musteri No'],
        'KACC Musteri Adı': kacc_ids['Musteri Adı'],
        'KACC Lat': k_lat,
        'KACC Lon': k_lon,
        'POS Musteri No': pos_ids['Musteri No'].to_numpy()[argmin],
        'POS Musteri Adı': pos_ids['Musteri Adı'].to_numpy()[argmin],
        'POS Lat': p_lat[argmin],
        'POS Lon': p_lon[argmin],
        'Mesafe (m)': np.round(min_dist, 0).astype(int),
    })
    if logger:
        logger(f"Sonuç hazır: {len(res)} satır")
    return res


def compute_within_radius(
    df_kacc: pd.DataFrame,
    df_pos: pd.DataFrame,
    meters: float,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")
        logger(f"Radius araması başlıyor: {meters} m")
    # Precompute for POS arrays
    p_lat_arr = p_lat
    p_lon_arr = p_lon

    rows = []

    # Precompute for POS arrays
    p_lat_arr = p_lat
    p_lon_arr = p_lon

    rows = []
    # Degree deltas per meter (approx)
    # delta_lat ≈ meters / 111_320
    # delta_lon ≈ meters / (111_320 * cos(lat))
    for i in range(len(k_lat)):
        lat = k_lat[i]
        lon = k_lon[i]
        dlat = meters / 111_320.0
        # Avoid division by zero for cos near poles; clamp cos
        cos_lat = np.cos(np.deg2rad(lat))
        cos_lat = cos_lat if abs(cos_lat) > 1e-6 else 1e-6
        dlon = meters / (111_320.0 * cos_lat)

        lat_min = lat - dlat
        lat_max = lat + dlat
        lon_min = lon - dlon
        lon_max = lon + dlon

        mask = (p_lat_arr >= lat_min) & (p_lat_arr <= lat_max) & (p_lon_arr >= lon_min) & (p_lon_arr <= lon_max)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        cand_lat = p_lat_arr[idx]
        cand_lon = p_lon_arr[idx]
        # Compute Haversine distances to candidates
        # We compute vector distances to the candidate set
        dist_vec = haversine_matrix(np.array([lat]), np.array([lon]), cand_lat, cand_lon)[0]
        sel = np.where(dist_vec <= meters)[0]
        if sel.size == 0:
            continue

        sel_idx = idx[sel]
        for j, d in zip(sel_idx, dist_vec[sel]):
            rows.append({
                'KACC Musteri No': kacc_ids.iloc[i]['Musteri No'],
                'KACC Musteri Adı': kacc_ids.iloc[i]['Musteri Adı'],
                'KACC Lat': lat,
                'KACC Lon': lon,
                'POS Musteri No': pos_ids.iloc[j]['Musteri No'],
                'POS Musteri Adı': pos_ids.iloc[j]['Musteri Adı'],
                'POS Lat': p_lat_arr[j],
                'POS Lon': p_lon_arr[j],
                'Mesafe (m)': int(np.round(d, 0)),
            })

        if logger and (i + 1) % 200 == 0:
            logger(f"İlerleme: {i + 1}/{len(k_lat)} KACC işlendi")

    res = pd.DataFrame(rows)
    return res


def write_output(output_path: str, df: pd.DataFrame, sheet_name: str) -> None:
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        log(f"Hata: çıktı yazılamadı: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='KACC müşterileri için POS hesaplama uygulaması (nearest veya radius).')
    parser.add_argument('--input', required=True, help='Giriş Excel dosyası (örn. Can karakaş Çalışma.xlsx)')
    parser.add_argument('--output', required=False, help='Çıktı Excel dosyası (varsayılan: input - nearest.xlsx)')
    parser.add_argument('--mode', choices=['nearest', 'radius'], default='nearest', help='Hesaplama modu: nearest (en yakın) veya radius (metre yarıçapı)')
    parser.add_argument('--meters', type=float, help='Radius modu için metre değeri (ör. 500)')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        log(f"Hata: dosya bulunamadı: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        dirn = os.path.dirname(input_path)
        output_path = os.path.join(dirn, f"{base} - nearest.xlsx")

    log(f"Excel okunuyor: {input_path}")
    df_kacc, df_pos = read_sheets(input_path)

    if args.mode == 'nearest':
        log("Mesafeler hesaplanıyor (Haversine, en yakın)")
        res = compute_nearest(df_kacc, df_pos)
        sheet_name = 'NearestPOS'
    else:
        if not args.meters or args.meters <= 0:
            log('Hata: radius modu için --meters pozitif bir sayı olmalı.')
            sys.exit(2)
        log(f"Mesafeler hesaplanıyor (Haversine, radius={args.meters} m)")
        res = compute_within_radius(df_kacc, df_pos, args.meters)
        sheet_name = 'POSWithinRadius'

    log(f"Sonuç satır sayısı: {len(res)}")
    log(f"Çıktı yazılıyor: {output_path}")
    write_output(output_path, res, sheet_name)

    log("Tamamlandı.")


if __name__ == '__main__':
    main()

