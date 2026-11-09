# encoding: utf-8
"""
CheXNet Inference & Localization Engine
Handles model loading, forward pass, multi-label thoracic disease classification,
and Class Activation Mapping (CAM) localization.
"""

import os
import re
import io
import time
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import DenseNet121

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

CLASS_META = {
    'Atelectasis': {
        'display': 'Atelectasis',
        'category': 'Pulmonary Volume',
        'desc': 'Partial or complete collapse of the entire lung or lobe of the lung.',
        'benchmark_auroc': 0.8294
    },
    'Cardiomegaly': {
        'display': 'Cardiomegaly',
        'category': 'Cardiac',
        'desc': 'Enlarged cardiac silhouette with cardiothoracic ratio exceeding 0.50.',
        'benchmark_auroc': 0.9165
    },
    'Effusion': {
        'display': 'Pleural Effusion',
        'category': 'Pleural Space',
        'desc': 'Abnormal fluid accumulation between the parietal and visceral pleura.',
        'benchmark_auroc': 0.8870
    },
    'Infiltration': {
        'display': 'Infiltration',
        'category': 'Parenchymal',
        'desc': 'Ill-defined pulmonary opacification representing fluid, blood, or pus.',
        'benchmark_auroc': 0.7143
    },
    'Mass': {
        'display': 'Mass (>3cm)',
        'category': 'Neoplastic / Solitary',
        'desc': 'Solitary pulmonary lesion greater than 3 cm in greatest dimension.',
        'benchmark_auroc': 0.8597
    },
    'Nodule': {
        'display': 'Nodule (<=3cm)',
        'category': 'Neoplastic / Solitary',
        'desc': 'Small, well-defined pulmonary parenchymal opacity 3 cm or less.',
        'benchmark_auroc': 0.7873
    },
    'Pneumonia': {
        'display': 'Pneumonia',
        'category': 'Infectious',
        'desc': 'Inflammatory infection of pulmonary alveoli producing exudative consolidation.',
        'benchmark_auroc': 0.7745
    },
    'Pneumothorax': {
        'display': 'Pneumothorax',
        'category': 'Pleural Space',
        'desc': 'Abnormal presence of air in the pleural cavity causing partial lung collapse.',
        'benchmark_auroc': 0.8726
    },
    'Consolidation': {
        'display': 'Consolidation',
        'category': 'Alveolar',
        'desc': 'Alveolar air spaces replaced by inflammatory liquid, blood, or cells.',
        'benchmark_auroc': 0.8142
    },
    'Edema': {
        'display': 'Pulmonary Edema',
        'category': 'Vascular / Fluid',
        'desc': 'Excess interstitial or alveolar fluid accumulation from heart failure or ARDS.',
        'benchmark_auroc': 0.8932
    },
    'Emphysema': {
        'display': 'Emphysema',
        'category': 'Obstructive',
        'desc': 'Hyperinflation and permanent enlargement of air spaces distal to terminal bronchioles.',
        'benchmark_auroc': 0.9254
    },
    'Fibrosis': {
        'display': 'Pulmonary Fibrosis',
        'category': 'Interstitial',
        'desc': 'Chronic scarring and thickening of parenchymal interstitial tissue.',
        'benchmark_auroc': 0.8304
    },
    'Pleural_Thickening': {
        'display': 'Pleural Thickening',
        'category': 'Pleural Space',
        'desc': 'Fibrous cicatrization and thickening of the pleura from prior inflammation.',
        'benchmark_auroc': 0.7831
    },
    'Hernia': {
        'display': 'Diaphragmatic Hernia',
        'category': 'Diaphragmatic',
        'desc': 'Protrusion of abdominal viscera through diaphragm into the thoracic cavity.',
        'benchmark_auroc': 0.9104
    }
}

COLORMAP_DICT = {
    'jet': cv2.COLORMAP_JET,
    'turbo': cv2.COLORMAP_TURBO,
    'inferno': cv2.COLORMAP_INFERNO,
    'plasma': cv2.COLORMAP_PLASMA,
    'viridis': cv2.COLORMAP_VIRIDIS
}


class CheXNetEngine:
    def __init__(self, ckpt_path='model.pth.tar', device=None):
        self.ckpt_path = ckpt_path
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.weights = None # classifier linear weights [14, 1024]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # In-memory feature cache for fast CAM re-computation
        self.last_cache = None
        self._load_model()

    def _load_model(self):
        print(f"[CheXNetEngine] Initializing model on {self.device}...")
        model = DenseNet121(len(CLASS_NAMES))
        
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.ckpt_path}")
            
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Compatibility regex for torchvision densenet naming change
        # (e.g. norm.1 -> norm1, conv.1 -> conv1)
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|conv))\.((?:[12])\..*)$')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            k = pattern.sub(r'\1\2', k)
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        
        self.model = model
        # Linear layer weight matrix: shape [14, 1024]
        self.weights = model.densenet121.classifier[0].weight.data.clone()
        print(f"[CheXNetEngine] Model successfully loaded ({sum(p.numel() for p in model.parameters()):,} parameters).")

    def _pil_to_base64(self, pil_img, format='PNG'):
        buf = io.BytesIO()
        pil_img.save(buf, format=format)
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

    def _cv2_to_base64(self, cv2_bgr, ext='.png'):
        success, encoded = cv2.imencode(ext, cv2_bgr)
        if not success:
            return ''
        return 'data:image/png;base64,' + base64.b64encode(encoded.tobytes()).decode('utf-8')

    def compute_cam(self, features, orig_bgr, class_idx, colormap='jet', alpha=0.45):
        """
        Computes Class Activation Map (CAM) for a given class index.
        features: [1, 1024, 7, 7] tensor
        orig_bgr: numpy array of original image
        """
        orig_h, orig_w = orig_bgr.shape[:2]
        class_weights = self.weights[class_idx].to(self.device) # [1024]
        
        # Weighted sum across 1024 feature channels
        cam = torch.zeros(features.shape[2:], dtype=torch.float32, device=self.device)
        for i, w_val in enumerate(class_weights):
            cam += w_val * features[0, i, :, :]
            
        cam = F.relu(cam)
        cam_np = cam.cpu().detach().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)
            
        # Resize CAM to original image dimensions
        cam_resized = cv2.resize(cam_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_u8 = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        cv_cmap = COLORMAP_DICT.get(colormap.lower(), cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(cam_u8, cv_cmap)
        
        # Overlay with original image
        overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
        
        # Calculate peak activation / hotspot bounding box
        # Peak point
        peak_idx = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
        peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
        
        # Extract hotspot region contour
        thresh_val = int(255 * 0.65) # top 35% activation region
        _, thresh = cv2.threshold(cam_u8, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                bx, by, bw, bh = cv2.boundingRect(largest)
                box = {
                    'x': int(bx),
                    'y': int(by),
                    'w': int(bw),
                    'h': int(bh),
                    'x_pct': round(bx / orig_w * 100, 2),
                    'y_pct': round(by / orig_h * 100, 2),
                    'w_pct': round(bw / orig_w * 100, 2),
                    'h_pct': round(bh / orig_h * 100, 2)
                }

        # Draw hotspot box on a copy for visual confirmation
        overlay_with_box = overlay.copy()
        if box:
            cv2.rectangle(
                overlay_with_box,
                (box['x'], box['y']),
                (box['x'] + box['w'], box['y'] + box['h']),
                (0, 240, 255), 2
            )
            cv2.circle(overlay_with_box, (peak_x, peak_y), 5, (0, 0, 255), -1)

        return {
            'cam_norm': cam_resized,
            'heatmap_bgr': heatmap_colored,
            'overlay_bgr': overlay,
            'overlay_with_box_bgr': overlay_with_box,
            'peak': {'x': peak_x, 'y': peak_y, 'intensity': float(cam_norm.max())},
            'box': box
        }

    def validate_chest_xray(self, pil_image):
        """
        Validates whether an input image is a genuine frontal chest (lung) X-ray.
        Rejects non-chest images (color photos, natural scenes, documents, extremities, etc.)
        Returns (is_valid: bool, message: str)
        """
        w, h = pil_image.size
        
        # 1. Aspect ratio check
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.55 or aspect_ratio > 1.75:
            return False, "This sample is not acceptable. The aspect ratio does not match a frontal thoracic chest projection."
            
        if w < 120 or h < 120:
            return False, "This sample is not acceptable. Image resolution is too low for radiological analysis."
            
        # Convert to OpenCV numpy formats
        np_img = np.array(pil_image)
        if len(np_img.shape) == 2:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        elif np_img.shape[2] == 4:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # 2. Color saturation test
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mean_sat = float(sat.mean())
        saturated_ratio = float((sat > 40).mean())
        
        # RGB channel divergence
        b, g, r = cv2.split(img_bgr)
        rgb_diff = (np.abs(b.astype(int) - g.astype(int)) + 
                    np.abs(g.astype(int) - r.astype(int)) + 
                    np.abs(r.astype(int) - b.astype(int))) / 3.0
        mean_rgb_diff = float(rgb_diff.mean())

        if mean_sat > 22.0 or saturated_ratio > 0.06 or mean_rgb_diff > 20.0:
            return False, "This sample is not acceptable. Only monochrome frontal chest (lung) radiographs are accepted. Color images are not valid."

        # 3. Grayscale intensity distribution & contrast
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        std_val = float(gray.std())
        
        if std_val < 12.0:
            return False, "This sample is not acceptable. The image is blank or lacks diagnostic radiographic contrast."
            
        extreme_white = float((gray > 240).mean())
        extreme_black = float((gray < 15).mean())
        if extreme_white > 0.60:
            return False, "This sample is not acceptable. The image appears to be a text document or graphic rather than a chest X-ray."

        # 4. Thoracic Anatomical Landmark & Lung Silhouette Analysis
        r256 = cv2.resize(gray, (256, 256)).astype(np.float32)
        
        left_lung = float(r256[70:170, 40:96].mean())
        right_lung = float(r256[70:170, 160:216].mean())
        center_mediastinum = float(r256[80:190, 100:156].mean())
        lung_mean = (left_lung + right_lung) / 2.0
        
        med_contrast = center_mediastinum - lung_mean
        symm_diff = abs(left_lung - right_lung) / (lung_mean + 1e-5)
        corner_mean = float((r256[:25, :25].mean() + r256[:25, -25:].mean()) / 2.0)
        
        # Bone/extremity check: black background (>65% black) with bone in center
        if extreme_black > 0.65 and corner_mean < 8.0:
            return False, "This sample is not acceptable. The image appears to be an extremity (bone/joint) X-ray rather than a chest radiograph."
            
        # Check for presence of thoracic anatomy
        if -6.0 < med_contrast < 6.0:
            return False, "This sample is not acceptable. The image does not exhibit anatomical thoracic lung fields and cardiac silhouette."
            
        if symm_diff > 0.38:
            return False, "This sample is not acceptable. Asymmetric bilateral profile does not correspond to a frontal chest radiograph."

        return True, "Valid frontal chest X-ray radiograph."

    def predict(self, pil_image, target_disease_idx=None, colormap='jet', alpha=0.45):
        """
        Runs CheXNet inference on a PIL image.
        First validates that the image is a genuine frontal chest (lung) X-ray.
        If non-chest image is given, rejects with a short explanation note.
        """
        start_time = time.time()
        orig_rgb = pil_image.convert('RGB')
        
        # Perform thoracic chest X-ray validation
        is_valid, validation_note = self.validate_chest_xray(orig_rgb)
        if not is_valid:
            orig_bgr = cv2.cvtColor(np.array(orig_rgb), cv2.COLOR_RGB2BGR)
            return {
                'is_valid_chest_xray': False,
                'rejection_note': validation_note,
                'status': 'rejected',
                'images': {
                    'original': self._cv2_to_base64(orig_bgr)
                }
            }

        orig_w, orig_h = orig_rgb.size
        orig_bgr = cv2.cvtColor(np.array(orig_rgb), cv2.COLOR_RGB2BGR)
        
        # Prepare tensor
        inp_tensor = self.transform(orig_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.densenet121.features(inp_tensor)
            features = F.relu(features, inplace=True)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            logits = self.model.densenet121.classifier[0](pooled)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        latency_ms = round((time.time() - start_time) * 1000, 1)
        
        # Determine disease rankings
        disease_results = []
        for idx, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            p = float(prob)
            meta = CLASS_META.get(name, {})
            
            if p >= 0.50:
                severity = 'High Risk'
                badge_class = 'badge-danger'
            elif p >= 0.20:
                severity = 'Moderate'
                badge_class = 'badge-warning'
            else:
                severity = 'Low / Normal'
                badge_class = 'badge-success'

            disease_results.append({
                'index': idx,
                'name': name,
                'display': meta.get('display', name),
                'category': meta.get('category', 'Thoracic'),
                'description': meta.get('desc', ''),
                'benchmark_auroc': meta.get('benchmark_auroc', 0.8),
                'probability': round(p, 4),
                'percentage': round(p * 100, 1),
                'severity': severity,
                'badge_class': badge_class,
                'is_positive': bool(p >= 0.35)
            })

        # Sort by highest probability
        ranked = sorted(disease_results, key=lambda x: x['probability'], reverse=True)
        top_disease = ranked[0]
        
        if target_disease_idx is None or target_disease_idx < 0 or target_disease_idx >= len(CLASS_NAMES):
            active_disease_idx = top_disease['index']
        else:
            active_disease_idx = target_disease_idx

        # Compute CAM for the active disease
        cam_info = self.compute_cam(features, orig_bgr, active_disease_idx, colormap=colormap, alpha=alpha)
        
        # Cache for fast re-rendering
        self.last_cache = {
            'features': features,
            'orig_bgr': orig_bgr,
            'orig_w': orig_w,
            'orig_h': orig_h,
            'probs': probs,
            'timestamp': time.time()
        }

        # Generate Clinical Impressions Memo
        impressions = self._generate_report(ranked, orig_w, orig_h, latency_ms)

        return {
            'latency_ms': latency_ms,
            'image_dims': {'width': orig_w, 'height': orig_h},
            'top_disease': top_disease,
            'active_disease_idx': active_disease_idx,
            'active_disease_name': CLASS_NAMES[active_disease_idx],
            'diseases': ranked,
            'raw_probabilities': {d['name']: d['probability'] for d in disease_results},
            'hotspot': cam_info['box'],
            'peak': cam_info['peak'],
            'report': impressions,
            'images': {
                'original': self._cv2_to_base64(orig_bgr),
                'overlay': self._cv2_to_base64(cam_info['overlay_bgr']),
                'overlay_with_box': self._cv2_to_base64(cam_info['overlay_with_box_bgr']),
                'heatmap_only': self._cv2_to_base64(cam_info['heatmap_bgr'])
            }
        }

    def recompute_cam(self, disease_idx, colormap='jet', alpha=0.45):
        """
        Fast re-render of CAM when user selects another disease in the UI.
        Runs in ~5-10ms directly from cached feature tensor without model forward pass.
        """
        if self.last_cache is None:
            return None
            
        features = self.last_cache['features']
        orig_bgr = self.last_cache['orig_bgr']
        cam_info = self.compute_cam(features, orig_bgr, disease_idx, colormap=colormap, alpha=alpha)
        
        return {
            'active_disease_idx': disease_idx,
            'active_disease_name': CLASS_NAMES[disease_idx],
            'hotspot': cam_info['box'],
            'peak': cam_info['peak'],
            'images': {
                'overlay': self._cv2_to_base64(cam_info['overlay_bgr']),
                'overlay_with_box': self._cv2_to_base64(cam_info['overlay_with_box_bgr']),
                'heatmap_only': self._cv2_to_base64(cam_info['heatmap_bgr'])
            }
        }

    def _generate_report(self, ranked_diseases, w, h, latency_ms):
        """Generates formal radiology-style clinical report text."""
        findings = []
        positive_findings = [d for d in ranked_diseases if d['probability'] >= 0.35]
        moderate_findings = [d for d in ranked_diseases if 0.15 <= d['probability'] < 0.35]

        if positive_findings:
            top = positive_findings[0]
            findings.append(f"Primary radiological finding demonstrates significant likelihood of {top['display']} ({top['percentage']}% confidence).")
            if len(positive_findings) > 1:
                others = ", ".join([f"{d['display']} ({d['percentage']}%)" for d in positive_findings[1:]])
                findings.append(f"Concurrent secondary thoracic abnormalities detected: {others}.")
        else:
            findings.append("No prominent thoracic parenchymal or pleural abnormalities detected above positive risk threshold (>=35%).")

        if moderate_findings:
            borderline = ", ".join([f"{d['display']} ({d['percentage']}%)" for d in moderate_findings[:3]])
            findings.append(f"Equivocal or borderline indications noted for: {borderline}.")

        summary = " ".join(findings)
        
        return {
            'study_type': 'Chest Radiograph (PA/AP Projection)',
            'resolution': f'{w} x {h} px',
            'analysis_time': f'{latency_ms} ms',
            'primary_finding': positive_findings[0]['display'] if positive_findings else 'No acute pathology',
            'clinical_summary': summary,
            'recommendation': 'Clinical correlation and formal review by a licensed radiologist is recommended.'
        }
