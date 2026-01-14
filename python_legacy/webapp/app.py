import os
import sys
import threading
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_file, redirect, session
import tempfile
import shutil
from functools import wraps
import pandas as pd
import numpy as np


# Üst dizini import yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# İç fonksiyonlar nearest_pos modülünden alınır
from nearest_pos import read_sheets, write_output
from nearest_pos import compute_nearest as original_compute_nearest, compute_within_radius as original_compute_within_radius

# Helper functions with cancellation support

def compute_nearest_with_cancel_support(df_kacc, df_pos, logger=None, job=None, temp_dir=None):
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")
        logger("Mesafe matrisi hesaplanıyor")
    
    # Check for cancellation before expensive operation
    if job and job.get('cancelled'):
        # Create partial result with just the KACC data and empty POS fields
        cancelled_res = pd.DataFrame({
            'KACC Musteri No': kacc_ids['Musteri No'],
            'KACC Musteri Adı': kacc_ids['Musteri Adı'],
            'KACC Lat': k_lat,
            'KACC Lon': k_lon,
            'POS Musteri No': np.nan,
            'POS Musteri Adı': np.nan,
            'POS Lat': np.nan,
            'POS Lon': np.nan,
            'Mesafe (m)': np.nan,
        })
        job['status'] = 'cancelled'
        job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore cleanup errors
        return cancelled_res
        
    # Calculate distance matrix in chunks to allow cancellation checks
    R = 6371000.0
    to_rad = np.pi / 180.0
    
    lat1r = k_lat[:, None] * to_rad
    lon1r = k_lon[:, None] * to_rad
    lat2r = p_lat[None, :] * to_rad
    lon2r = p_lon[None, :] * to_rad

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist = R * c
    
    # Check for cancellation after distance calculation
    if job and job.get('cancelled'):
        # Create partial result with just the KACC data and empty POS fields
        cancelled_res = pd.DataFrame({
            'KACC Musteri No': kacc_ids['Musteri No'],
            'KACC Musteri Adı': kacc_ids['Musteri Adı'],
            'KACC Lat': k_lat,
            'KACC Lon': k_lon,
            'POS Musteri No': np.nan,
            'POS Musteri Adı': np.nan,
            'POS Lat': np.nan,
            'POS Lon': np.nan,
            'Mesafe (m)': np.nan,
        })
        job['status'] = 'cancelled'
        job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore cleanup errors
        return cancelled_res

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


def compute_within_radius_with_cancel_support(df_kacc, df_pos, meters, logger=None, job=None, temp_dir=None):
    kacc_ids, k_lat, k_lon = extract_coords(df_kacc, 'KACC')
    pos_ids, p_lat, p_lon = extract_coords(df_pos, 'Pos')

    if logger:
        logger(f"KACC geçerli satır: {len(k_lat)} | POS geçerli satır: {len(p_lat)}")
        logger(f"Radius araması başlıyor: {meters} m")
    
    # Check for cancellation
    if job and job.get('cancelled'):
        # Create partial result with just the KACC data and empty POS fields
        cancelled_res = pd.DataFrame({
            'KACC Musteri No': kacc_ids['Musteri No'],
            'KACC Musteri Adı': kacc_ids['Musteri Adı'],
            'KACC Lat': k_lat,
            'KACC Lon': k_lon,
            'POS Musteri No': np.nan,
            'POS Musteri Adı': np.nan,
            'POS Lat': np.nan,
            'POS Lon': np.nan,
            'Mesafe (m)': np.nan,
        })
        job['status'] = 'cancelled'
        job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore cleanup errors
        return cancelled_res

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
            
            # Check for cancellation during processing
            if job and job.get('cancelled'):
                # Add all remaining KACC points as unprocessed
                remaining_rows = []
                for remaining_i in range(i + 1, len(k_lat)):
                    remaining_rows.append({
                        'KACC Musteri No': kacc_ids.iloc[remaining_i]['Musteri No'],
                        'KACC Musteri Adı': kacc_ids.iloc[remaining_i]['Musteri Adı'],
                        'KACC Lat': k_lat[remaining_i],
                        'KACC Lon': k_lon[remaining_i],
                        'POS Musteri No': np.nan,
                        'POS Musteri Adı': np.nan,
                        'POS Lat': np.nan,
                        'POS Lon': np.nan,
                        'Mesafe (m)': np.nan,
                    })
                
                # Combine processed results with unprocessed KACC points
                if rows:
                    processed_df = pd.DataFrame(rows)
                    remaining_df = pd.DataFrame(remaining_rows)
                    final_result = pd.concat([processed_df, remaining_df], ignore_index=True)
                else:
                    final_result = pd.DataFrame(remaining_rows)
                
                job['status'] = 'cancelled'
                job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass  # Ignore cleanup errors
                return final_result

    res = pd.DataFrame(rows)
    return res


def extract_coords(df, label):
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


def to_float_series(s):
    # Normalize decimal separators and coerce to float
    s = s.astype(str).str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def haversine_matrix(lat1, lon1, lat2, lon2):
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


app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-pos-distance-app'  # Secret key for sessions

DEFAULT_INPUT = os.path.join(os.getcwd(), 'Can karakaş Çalışma.xlsx')

# Login credentials
LOGIN_CREDENTIALS = {
    'username': 'user',
    'password': 'Dino202545'
}


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == LOGIN_CREDENTIALS['username'] and password == LOGIN_CREDENTIALS['password']:
            session['logged_in'] = True
            return redirect('/')
        else:
            return render_template('login.html', error="Geçersiz kullanıcı adı veya şifre")
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    return redirect('/login')


@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template(
        'index.html',
        default_input="",
        default_output="",
        message=None,
        result=None,
        logs=None,
        job_id=None,
    )


JOBS = {}


@app.route('/cancel', methods=['POST'])
def cancel():
    job_id = request.args.get('job_id')
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'job bulunamadı'}), 404
    
    job['cancelled'] = True
    job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
    
    return jsonify({'ok': True})


@app.route('/download-template', methods=['GET'])
@login_required
def download_template():
    template_path = os.path.join(BASE_DIR, 'template.xlsx')
    if os.path.exists(template_path):
        return send_file(template_path, as_attachment=True, download_name='template.xlsx')
    else:
        return "Template file not found", 404


@app.route('/download-result/<filename>', methods=['GET'])
@login_required
def download_result(filename):
    # Security check to prevent directory traversal
    import urllib.parse
    filename = urllib.parse.unquote(filename)
    safe_filename = os.path.basename(filename)
    output_path = os.path.join(os.getcwd(), safe_filename)
    
    if os.path.exists(output_path) and safe_filename == os.path.basename(output_path):
        return send_file(output_path, as_attachment=True, download_name=safe_filename)
    else:
        return "Result file not found", 404


def default_output_for(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    dirn = os.path.dirname(path)
    return os.path.join(dirn, f"{base} - nearest.xlsx")


@app.route('/run', methods=['POST'])
def run():
    # Handle file upload
    uploaded_file = request.files.get('input_file')
    if not uploaded_file or uploaded_file.filename == '':
        return "Lütfen bir Excel dosyası seçin", 400
    
    # Validate file extension
    if not uploaded_file.filename.lower().endswith(('.xlsx', '.xls')):
        return "Lütfen sadece Excel dosyalarını (.xlsx, .xls) yükleyin", 400
    
    # Save uploaded file to temporary location
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.filename)
    uploaded_file.save(temp_file_path)
    
    # Generate a unique output filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_output_{timestamp}.xlsx"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    mode = request.form.get('mode') or 'nearest'
    meters_str = request.form.get('meters') or ''

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'logs': [f'İş başlatıldı: {time.strftime("%H:%M:%S")}'],
        'status': 'running',
        'result': None,
        'error': None,
        'last_update': time.time(),
        'estimated_time': None,
        'start_time': time.time(),
        'temp_dir': temp_dir,  # Store temp directory to clean up later
    }

    t = threading.Thread(target=run_job, args=(job_id, temp_file_path, output_path, mode, meters_str, temp_dir), daemon=True)
    t.start()

    return render_template(
        'index.html',
        default_input="",
        default_output="",
        message='İşlem başladı. Loglar aşağıda canlı olarak güncellenecek.',
        result=None,
        logs=[],
        job_id=job_id,
    )


def run_job(job_id: str, input_path: str, output_path: str, mode: str, meters_str: str, temp_dir: str):
    job = JOBS[job_id]
    start_time = time.time()

    def logger(msg: str):
        timestamp = time.strftime("%H:%M:%S")
        job['logs'].append(f"[{timestamp}] {msg}")
        # Update progress info
        job['last_update'] = time.time()
        
        # Check for cancellation after logging
        if job.get('cancelled'):
            job['status'] = 'cancelled'
            job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
            # Clean up temp directory after cancellation
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors
            return True  # Indicate that the job was cancelled
        return False

    logger(f"Girdi: {os.path.basename(input_path)}")
    df_kacc, df_pos = read_sheets(input_path)
    
    # Calculate initial info for estimation
    kacc_count = len(df_kacc)
    pos_count = len(df_pos)
    logger(f"KACC satır sayısı: {kacc_count}")
    logger(f"POS satır sayısı: {pos_count}")
    
    # Check for cancellation early
    if job.get('cancelled'):
        job['status'] = 'cancelled'
        job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
        try:
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore cleanup errors
        return
    
    if mode == 'nearest':
        logger("Mod: nearest")
        logger("En yakın konumlar hesaplanıyor...")
        # For nearest, we need to calculate distances for each KACC to all POS points
        expected_calculations = kacc_count * pos_count
        logger(f"Beklenen yaklaşık hesaplama sayısı: {expected_calculations:,}")
        
        # Create a custom logger that updates progress
        def progress_logger(msg: str):
            timestamp = time.strftime("%H:%M:%S")
            job['logs'].append(f"[{timestamp}] {msg}")
            job['last_update'] = time.time()
            
            # Calculate progress based on KACC processing
            if "En yakın indeksler bulunuyor" in msg:
                job['progress'] = 90  # Almost done
            elif "Mesafe matrisi hesaplanıyor" in msg:
                job['progress'] = 50  # Halfway through heavy computation
            else:
                job['progress'] = 10  # Initial processing
            
            # Check for cancellation
            if job.get('cancelled'):
                job['status'] = 'cancelled'
                job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
                # Clean up temp directory after cancellation
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass  # Ignore cleanup errors
                return True  # Indicate that the job was cancelled
            return False
        
        res = compute_nearest_with_cancel_support(df_kacc, df_pos, logger=progress_logger, job=job, temp_dir=temp_dir)
        if job.get('cancelled'):
            # Result already contains all KACC points with empty POS data
            sheet_name = 'Cancelled_NearestPOS'
        else:
            sheet_name = 'NearestPOS'
    else:
        try:
            meters = float(meters_str)
        except Exception:
            meters = 0.0
        if meters <= 0:
            job['status'] = 'error'
            job['error'] = 'Lütfen radius modu için pozitif bir metre değeri giriniz.'
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors
            return
        logger(f"Mod: radius, meters={meters}")
        logger("Yarıçap içindeki konumlar hesaplanıyor...")
        
        # For radius mode, also add progress tracking
        def radius_progress_logger(msg: str):
            timestamp = time.strftime("%H:%M:%S")
            job['logs'].append(f"[{timestamp}] {msg}")
            job['last_update'] = time.time()
            
            if "radius araması başlıyor" in msg:
                job['progress'] = 20
            elif "İlerleme:" in msg:
                job['progress'] = 70  # Processing
            else:
                job['progress'] = 10
            
            # Check for cancellation
            if job.get('cancelled'):
                job['status'] = 'cancelled'
                job['logs'].append(f'[İptal] [{time.strftime("%H:%M:%S")}] İşlem kullanıcı tarafından durduruldu')
                # Clean up temp directory after cancellation
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass  # Ignore cleanup errors
                return True  # Indicate that the job was cancelled
            return False
        
        res = compute_within_radius_with_cancel_support(df_kacc, df_pos, meters, logger=radius_progress_logger, job=job, temp_dir=temp_dir)
        if job.get('cancelled'):
            # Result already contains processed and unprocessed KACC points
            sheet_name = 'Cancelled_POSWithinRadius'
        else:
            sheet_name = 'POSWithinRadius'

    logger(f"Sonuç satır: {len(res)}")
    logger(f"Yazılıyor: {output_path} ({sheet_name})")
    write_output(output_path, res, sheet_name)
    job['result'] = {
        'mode': mode,
        'rows': len(res),
        'output': output_path,
        'sheet': sheet_name,
    }
    job['status'] = 'done'
    end_time = time.time()
    job['processing_time'] = end_time - start_time
    logger(f"Tamamlandı. İşlem süresi: {job['processing_time']:.2f} saniye")
    
    # Clean up temp directory after processing
    try:
        shutil.rmtree(temp_dir)
    except:
        pass  # Ignore cleanup errors


@app.route('/logs', methods=['GET'])
def logs():
    job_id = request.args.get('job_id')
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'job bulunamadı'}), 404
    
    current_time = time.time()
    elapsed_time = current_time - job.get('start_time', current_time)
    
    # Calculate estimated time if possible
    estimated_time = None
    progress = job.get('progress', 0)
    
    if progress > 0 and progress < 100:
        # Estimate based on progress and elapsed time
        total_estimated_time = elapsed_time * 100 / progress
        remaining_time = total_estimated_time - elapsed_time
        estimated_time = f"Yaklaşık {remaining_time:.1f} saniye ({progress}% tamamlandı)"
    
    # Return logs with additional status info
    response_data = {
        'ok': True, 
        'logs': job['logs'], 
        'status': job['status'],
        'estimated_time': estimated_time,
        'elapsed_time': elapsed_time,
        'progress': progress
    }
    
    return jsonify(response_data)


@app.route('/status', methods=['GET'])
def status():
    job_id = request.args.get('job_id')
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'job bulunamadı'}), 404
    return jsonify({'ok': True, 'status': job['status'], 'result': job['result'], 'error': job['error']})


if __name__ == '__main__':
    print('Running on http://localhost:9595/')
    app.run(host='0.0.0.0', port=9595)
