# encoding: utf-8
"""
CheXNet Medical AI Web Application Backend
Flask server providing REST API for thoracic disease detection and CAM localization.
"""

import os
import io
import time
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
from inference_engine import CheXNetEngine, CLASS_NAMES, CLASS_META, COLORMAP_DICT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Initialize CheXNet Engine
engine = CheXNetEngine(ckpt_path=os.path.join(BASE_DIR, 'model.pth.tar'))

SAMPLE_METADATA = [
    {
        'id': 'sample_5_nih_patient.png',
        'title': 'Mass, Consolidation & Effusion',
        'category': 'NIH ChestX-ray14 Test Set',
        'description': 'Image 00000920_001.png. NIH labels: Mass, Consolidation, Effusion.',
        'path': os.path.join(SAMPLES_DIR, 'sample_5_nih_patient.png')
    },
    {
        'id': 'sample_cardiomegaly.png',
        'title': 'Cardiomegaly',
        'category': 'NIH ChestX-ray14 Test Set',
        'description': 'Image 00000864_000.png. NIH label: Cardiomegaly.',
        'path': os.path.join(SAMPLES_DIR, 'sample_cardiomegaly.png')
    },
    {
        'id': 'sample_pneumonia.png',
        'title': 'Pneumonia',
        'category': 'NIH ChestX-ray14 Test Set',
        'description': 'Image 00001021_000.png. NIH label: Pneumonia.',
        'path': os.path.join(SAMPLES_DIR, 'sample_pneumonia.png')
    },
    {
        'id': 'sample_effusion.png',
        'title': 'Pleural Effusion',
        'category': 'NIH ChestX-ray14 Test Set',
        'description': 'Image 00000402_009.png. NIH label: Effusion.',
        'path': os.path.join(SAMPLES_DIR, 'sample_effusion.png')
    },
    {
        'id': 'sample_normal.png',
        'title': 'No Finding',
        'category': 'NIH ChestX-ray14 Test Set',
        'description': 'Image 00001215_000.png. NIH label: No Finding.',
        'path': os.path.join(SAMPLES_DIR, 'sample_normal.png')
    }
]


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route('/samples/<filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route('/api/info', methods=['GET'])
def get_info():
    """Returns model architecture information, hardware status, and benchmark AUROC stats."""
    device_name = str(engine.device)
    total_params = sum(p.numel() for p in engine.model.parameters())
    
    # AUROC benchmark comparisons from the Stanford & PKU paper
    paper_benchmarks = [
        {'disease': 'Atelectasis', 'chexnet': 0.8094, 'our_impl': 0.8294, 'improved': 0.8311},
        {'disease': 'Cardiomegaly', 'chexnet': 0.9248, 'our_impl': 0.9165, 'improved': 0.9220},
        {'disease': 'Effusion', 'chexnet': 0.8638, 'our_impl': 0.8870, 'improved': 0.8891},
        {'disease': 'Infiltration', 'chexnet': 0.7345, 'our_impl': 0.7143, 'improved': 0.7146},
        {'disease': 'Mass', 'chexnet': 0.8676, 'our_impl': 0.8597, 'improved': 0.8627},
        {'disease': 'Nodule', 'chexnet': 0.7802, 'our_impl': 0.7873, 'improved': 0.7883},
        {'disease': 'Pneumonia', 'chexnet': 0.7680, 'our_impl': 0.7745, 'improved': 0.7820},
        {'disease': 'Pneumothorax', 'chexnet': 0.8887, 'our_impl': 0.8726, 'improved': 0.8844},
        {'disease': 'Consolidation', 'chexnet': 0.7901, 'our_impl': 0.8142, 'improved': 0.8148},
        {'disease': 'Edema', 'chexnet': 0.8878, 'our_impl': 0.8932, 'improved': 0.8992},
        {'disease': 'Emphysema', 'chexnet': 0.9371, 'our_impl': 0.9254, 'improved': 0.9343},
        {'disease': 'Fibrosis', 'chexnet': 0.8047, 'our_impl': 0.8304, 'improved': 0.8385},
        {'disease': 'Pleural Thickening', 'chexnet': 0.8062, 'our_impl': 0.7831, 'improved': 0.7914},
        {'disease': 'Hernia', 'chexnet': 0.9164, 'our_impl': 0.9104, 'improved': 0.9206}
    ]

    mean_chexnet = round(sum(b['chexnet'] for b in paper_benchmarks) / len(paper_benchmarks), 4)
    mean_our = round(sum(b['our_impl'] for b in paper_benchmarks) / len(paper_benchmarks), 4)

    return jsonify({
        'status': 'online',
        'model': {
            'name': 'CheXNet',
            'architecture': 'DenseNet-121 (Huang et al., modified with Sigmoid multi-label output)',
            'parameters': total_params,
            'input_resolution': '224 x 224 x 3 (RGB standard clinical normalization)',
            'classes_count': len(CLASS_NAMES),
            'classes': CLASS_NAMES,
            'device': device_name,
            'mean_auroc_chexnet': mean_chexnet,
            'mean_auroc_our_impl': mean_our
        },
        'benchmarks': paper_benchmarks,
        'colormaps': list(COLORMAP_DICT.keys())
    })


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Returns available sample radiographs."""
    samples_list = []
    for s in SAMPLE_METADATA:
        if os.path.exists(s['path']):
            samples_list.append({
                'id': s['id'],
                'title': s['title'],
                'category': s['category'],
                'description': s['description'],
                'url': f'/samples/{s["id"]}'
            })
    return jsonify({'samples': samples_list})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts:
    - 'file': uploaded image file OR
    - 'sample_id': ID of preset sample
    - 'disease_idx': target class for CAM localization (optional)
    - 'colormap': colormap name ('jet', 'turbo', 'inferno', 'plasma', 'viridis')
    - 'alpha': float blend ratio (default 0.45)
    """
    try:
        colormap = request.form.get('colormap', 'jet')
        alpha = float(request.form.get('alpha', 0.45))
        disease_idx_str = request.form.get('disease_idx')
        disease_idx = int(disease_idx_str) if disease_idx_str is not None and disease_idx_str != '' else None

        pil_img = None

        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file = request.files['file']
            pil_img = Image.open(uploaded_file.stream)
        elif request.form.get('sample_id'):
            sample_id = request.form.get('sample_id')
            sample_path = os.path.join(SAMPLES_DIR, sample_id)
            if not os.path.exists(sample_path):
                return jsonify({'error': f'Sample not found: {sample_id}'}), 404
            pil_img = Image.open(sample_path)
        else:
            return jsonify({'error': 'No image file or sample_id provided.'}), 400

        result = engine.predict(
            pil_img,
            target_disease_idx=disease_idx,
            colormap=colormap,
            alpha=alpha
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/recompute_cam', methods=['POST'])
def recompute_cam():
    """
    Re-renders CAM overlay for a specific disease without running full model forward pass.
    """
    try:
        data = request.get_json(force=True)
        disease_idx = int(data.get('disease_idx', 0))
        colormap = data.get('colormap', 'jet')
        alpha = float(data.get('alpha', 0.45))

        cam_res = engine.recompute_cam(disease_idx, colormap=colormap, alpha=alpha)
        if cam_res is None:
            return jsonify({'error': 'No image has been analyzed yet.'}), 400

        return jsonify(cam_res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """
    Runs diagnostic stress and latency test across sample radiographs.
    Calculates average latency, FPS throughput, and prediction consistency.
    """
    try:
        trials = []
        start_bench = time.time()
        
        # Test on 3 diverse sample images
        sample_files = [s['path'] for s in SAMPLE_METADATA if os.path.exists(s['path'])][:3]
        if not sample_files:
            return jsonify({'error': 'No sample files available for benchmark.'}), 400

        iterations = 5
        total_runs = 0
        latencies = []

        for sample_path in sample_files:
            img = Image.open(sample_path)
            sample_latencies = []
            sample_name = os.path.basename(sample_path)
            
            for it in range(iterations):
                t0 = time.time()
                res = engine.predict(img)
                t1 = time.time()
                dt_ms = round((t1 - t0) * 1000, 2)
                sample_latencies.append(dt_ms)
                latencies.append(dt_ms)
                total_runs += 1

            trials.append({
                'sample': sample_name,
                'top_prediction': res['top_disease']['display'],
                'top_confidence': f"{res['top_disease']['percentage']}%",
                'min_ms': round(min(sample_latencies), 1),
                'max_ms': round(max(sample_latencies), 1),
                'avg_ms': round(sum(sample_latencies) / len(sample_latencies), 1)
            })

        total_time = round(time.time() - start_bench, 2)
        avg_latency = round(sum(latencies) / len(latencies), 1)
        fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0

        return jsonify({
            'status': 'completed',
            'device': str(engine.device),
            'total_inferences': total_runs,
            'total_time_seconds': total_time,
            'average_latency_ms': avg_latency,
            'min_latency_ms': round(min(latencies), 1),
            'max_latency_ms': round(max(latencies), 1),
            'throughput_fps': fps,
            'trials': trials
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting CheXNet Web Application on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
