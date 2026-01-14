import os
import sys
import threading
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_file, redirect, session
import tempfile
import shutil
from functools import wraps


# Üst dizini import yoluna ekle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# İç fonksiyonlar nearest_pos modülünden alınır
from nearest_pos import read_sheets, compute_nearest, compute_within_radius, write_output

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

    logger(f"Girdi: {os.path.basename(input_path)}")
    df_kacc, df_pos = read_sheets(input_path)
    
    # Calculate initial info for estimation
    kacc_count = len(df_kacc)
    pos_count = len(df_pos)
    logger(f"KACC satır sayısı: {kacc_count}")
    logger(f"POS satır sayısı: {pos_count}")
    
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
        
        res = compute_nearest(df_kacc, df_pos, logger=progress_logger)
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
        
        res = compute_within_radius(df_kacc, df_pos, meters, logger=radius_progress_logger)
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
