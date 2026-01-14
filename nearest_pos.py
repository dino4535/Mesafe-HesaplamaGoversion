import argparse
import os
import sys
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd

# Opsiyonel: BallTree için scikit-learn
try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Opsiyonel: Numba JIT derleme
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def log(msg: str) -> None:
    print(msg, flush=True)


def to_float_series(s: pd.Series) -> pd.Series:
    """Normalize decimal separators and coerce to float."""
    s = s.astype(str).str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def read_sheets(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Excel dosyasından KACC ve Pos sayfalarını oku."""
    try:
        df_k = pd.read_excel(path, sheet_name='KACC', header=0)
        df_p = pd.read_excel(path, sheet_name='Pos', header=0)
    except Exception as e:
        log(f"Hata: Excel okuma başarısız: {e}")
        raise
    return df_k, df_p


def extract_coords(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """DataFrame'den koordinatları çıkar (J ve K sütunları)."""
    if df.shape[1] < 11:
        raise ValueError(f"{label} sayfasında en az 11 sütun bekleniyor (J ve K koordinatlar).")

    lat = to_float_series(df.iloc[:, 9])
    lon = to_float_series(df.iloc[:, 10])

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


# ============================================================================
# OPTIMIZE: Haversine hesaplama - Numba JIT ile hızlandırılmış
# ============================================================================

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def haversine_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Tek nokta çifti için Haversine mesafesi (metre)."""
        R = 6371000.0
        to_rad = np.pi / 180.0
        
        lat1r = lat1 * to_rad
        lon1r = lon1 * to_rad
        lat2r = lat2 * to_rad
        lon2r = lon2 * to_rad
        
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c

    @njit(parallel=True, fastmath=True)
    def haversine_to_all(lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
        """Bir noktadan tüm noktalara Haversine mesafesi."""
        n = len(lat2_arr)
        result = np.empty(n, dtype=np.float64)
        R = 6371000.0
        to_rad = np.pi / 180.0
        
        lat1r = lat1 * to_rad
        lon1r = lon1 * to_rad
        cos_lat1 = np.cos(lat1r)
        
        for i in prange(n):
            lat2r = lat2_arr[i] * to_rad
            lon2r = lon2_arr[i] * to_rad
            
            dlat = lat2r - lat1r
            dlon = lon2r - lon1r
            
            a = np.sin(dlat / 2.0) ** 2 + cos_lat1 * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            result[i] = R * c
        
        return result
else:
    def haversine_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Pure NumPy fallback."""
        R = 6371000.0
        to_rad = np.pi / 180.0
        lat1r, lon1r = lat1 * to_rad, lon1 * to_rad
        lat2r, lon2r = lat2 * to_rad, lon2 * to_rad
        dlat, dlon = lat2r - lat1r, lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c

    def haversine_to_all(lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
        """Pure NumPy fallback."""
        R = 6371000.0
        to_rad = np.pi / 180.0
        lat1r, lon1r = lat1 * to_rad, lon1 * to_rad
        lat2r = lat2_arr * to_rad
        lon2r = lon2_arr * to_rad
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Orijinal matris hesaplama (backward compatibility)."""
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


# ============================================================================
# OPTIMIZE: BallTree kullanarak O(N*M) -> O(N*log(M)) karmaşıklığı
# ============================================================================

def compute_nearest(
    df_kacc: pd.DataFrame,
    df_pos: pd.DataFrame,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """En yakın POS'u bul - BallTree ile optimize edilmiş."""
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")

    # BallTree varsa kullan (çok daha hızlı)
    if HAS_SKLEARN:
        if logger:
            logger("BallTree ile hızlı arama kullanılıyor")
        
        # Koordinatları radyana çevir
        pos_coords_rad = np.deg2rad(np.column_stack([p_lat, p_lon]))
        kacc_coords_rad = np.deg2rad(np.column_stack([k_lat, k_lon]))
        
        # BallTree oluştur
        tree = BallTree(pos_coords_rad, metric='haversine')
        
        if logger:
            logger("En yakın konumlar aranıyor...")
        
        # Batch halinde işle ve ilerleme göster
        total_kacc = len(k_lat)
        batch_size = 500
        all_distances = []
        all_indices = []
        
        for start_idx in range(0, total_kacc, batch_size):
            end_idx = min(start_idx + batch_size, total_kacc)
            batch_coords = kacc_coords_rad[start_idx:end_idx]
            
            # Bu batch için en yakın komşuyu bul
            batch_dist, batch_idx = tree.query(batch_coords, k=1)
            all_distances.append(batch_dist)
            all_indices.append(batch_idx)
            
            # İlerleme logla
            if logger:
                pct = (end_idx / total_kacc) * 100
                logger(f"İşlenen KACC: {end_idx} / {total_kacc} (%{pct:.1f})")
        
        # Sonuçları birleştir
        distances_rad = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        
        # Radyan mesafeyi metreye çevir (Dünya yarıçapı: 6371000 m)
        min_dist = distances_rad.flatten() * 6371000.0
        argmin = indices.flatten()
        
        if logger:
            logger("Tüm KACC bayileri işlendi ✓")
    else:
        # Fallback: Orijinal matris yöntemi
        if logger:
            logger("Mesafe matrisi hesaplanıyor (klasik yöntem)")
        dist = haversine_matrix(k_lat, k_lon, p_lat, p_lon)
        if logger:
            logger("En yakın indeksler bulunuyor")
        argmin = np.argmin(dist, axis=1)
        min_dist = dist[np.arange(dist.shape[0]), argmin]

    # Sonuç DataFrame'i oluştur
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


# ============================================================================
# OPTIMIZE: Vektörize edilmiş radius araması
# ============================================================================

def compute_within_radius(
    df_kacc: pd.DataFrame,
    df_pos: pd.DataFrame,
    meters: float,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Yarıçap içindeki POS'ları bul - Optimize edilmiş."""
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")

    # BallTree varsa kullan
    if HAS_SKLEARN:
        if logger:
            logger(f"BallTree ile radius araması: {meters} m")
        
        pos_coords_rad = np.deg2rad(np.column_stack([p_lat, p_lon]))
        kacc_coords_rad = np.deg2rad(np.column_stack([k_lat, k_lon]))
        
        tree = BallTree(pos_coords_rad, metric='haversine')
        
        # Metre -> radyan dönüşümü
        radius_rad = meters / 6371000.0
        
        # Radius içindeki tüm noktaları bul
        indices_list, distances_list = tree.query_radius(
            kacc_coords_rad, r=radius_rad, return_distance=True
        )
        
        # Sonuçları topla
        rows = []
        for i, (idx_arr, dist_arr) in enumerate(zip(indices_list, distances_list)):
            if len(idx_arr) == 0:
                continue
            
            # Radyan -> metre dönüşümü
            dist_meters = dist_arr * 6371000.0
            
            for j, d in zip(idx_arr, dist_meters):
                rows.append({
                    'KACC Musteri No': kacc_ids.iloc[i]['Musteri No'],
                    'KACC Musteri Adı': kacc_ids.iloc[i]['Musteri Adı'],
                    'KACC Lat': k_lat[i],
                    'KACC Lon': k_lon[i],
                    'POS Musteri No': pos_ids.iloc[j]['Musteri No'],
                    'POS Musteri Adı': pos_ids.iloc[j]['Musteri Adı'],
                    'POS Lat': p_lat[j],
                    'POS Lon': p_lon[j],
                    'Mesafe (m)': int(np.round(d, 0)),
                })
            
            if logger and (i + 1) % 500 == 0:
                logger(f"İlerleme: {i + 1}/{len(k_lat)} KACC işlendi")
        
        if logger:
            logger(f"Toplam {len(rows)} eşleşme bulundu")
    else:
        # Fallback: Optimize edilmiş klasik yöntem
        if logger:
            logger(f"Radius araması başlıyor: {meters} m")
        
        rows = []
        dlat_base = meters / 111_320.0
        
        for i in range(len(k_lat)):
            lat = k_lat[i]
            lon = k_lon[i]
            
            cos_lat = np.cos(np.deg2rad(lat))
            cos_lat = max(cos_lat, 1e-6)
            dlon = meters / (111_320.0 * cos_lat)

            lat_min, lat_max = lat - dlat_base, lat + dlat_base
            lon_min, lon_max = lon - dlon, lon + dlon

            # Vektörize filtreleme
            mask = (p_lat >= lat_min) & (p_lat <= lat_max) & (p_lon >= lon_min) & (p_lon <= lon_max)
            idx = np.where(mask)[0]
            
            if idx.size == 0:
                continue

            # Numba veya NumPy ile mesafe hesapla
            dist_vec = haversine_to_all(lat, lon, p_lat[idx], p_lon[idx])
            sel = dist_vec <= meters
            
            for j, d in zip(idx[sel], dist_vec[sel]):
                rows.append({
                    'KACC Musteri No': kacc_ids.iloc[i]['Musteri No'],
                    'KACC Musteri Adı': kacc_ids.iloc[i]['Musteri Adı'],
                    'KACC Lat': lat,
                    'KACC Lon': lon,
                    'POS Musteri No': pos_ids.iloc[j]['Musteri No'],
                    'POS Musteri Adı': pos_ids.iloc[j]['Musteri Adı'],
                    'POS Lat': p_lat[j],
                    'POS Lon': p_lon[j],
                    'Mesafe (m)': int(np.round(d, 0)),
                })

            if logger and (i + 1) % 200 == 0:
                logger(f"İlerleme: {i + 1}/{len(k_lat)} KACC işlendi")

    res = pd.DataFrame(rows)
    return res


def write_output(output_path: str, df: pd.DataFrame, sheet_name: str) -> None:
    """Sonucu Excel dosyasına yaz."""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        log(f"Hata: çıktı yazılamadı: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='KACC müşterileri için POS hesaplama uygulaması (optimize edilmiş).'
    )
    parser.add_argument('--input', required=True, help='Giriş Excel dosyası')
    parser.add_argument('--output', required=False, help='Çıktı Excel dosyası')
    parser.add_argument('--mode', choices=['nearest', 'radius'], default='nearest',
                        help='Hesaplama modu: nearest (en yakın) veya radius')
    parser.add_argument('--meters', type=float, help='Radius modu için metre değeri')
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

    # Kullanılan optimizasyonları göster
    log("=" * 50)
    log("POS Distance Calculator - Optimize Edilmiş Versiyon")
    log("=" * 50)
    log(f"BallTree (sklearn): {'✓ Aktif' if HAS_SKLEARN else '✗ Yüklü değil'}")
    log(f"Numba JIT: {'✓ Aktif' if HAS_NUMBA else '✗ Yüklü değil'}")
    log("=" * 50)

    log(f"Excel okunuyor: {input_path}")
    df_kacc, df_pos = read_sheets(input_path)

    if args.mode == 'nearest':
        log("Mesafeler hesaplanıyor (en yakın)")
        res = compute_nearest(df_kacc, df_pos, logger=log)
        sheet_name = 'NearestPOS'
    else:
        if not args.meters or args.meters <= 0:
            log('Hata: radius modu için --meters pozitif bir sayı olmalı.')
            sys.exit(2)
        log(f"Mesafeler hesaplanıyor (radius={args.meters} m)")
        res = compute_within_radius(df_kacc, df_pos, args.meters, logger=log)
        sheet_name = 'POSWithinRadius'

    log(f"Sonuç satır sayısı: {len(res)}")
    log(f"Çıktı yazılıyor: {output_path}")
    write_output(output_path, res, sheet_name)

    log("Tamamlandı.")


if __name__ == '__main__':
    main()
