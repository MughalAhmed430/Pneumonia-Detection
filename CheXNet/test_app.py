# encoding: utf-8
"""
CheXNet Test & Validation Suite
Automated test suite verifying model loading, multi-label prediction,
CAM localization, and HTTP REST API endpoints.
"""

import os
import unittest
import requests
from PIL import Image
from inference_engine import CheXNetEngine, CLASS_NAMES

BASE_URL = "http://127.0.0.1:5000"


class TestCheXNetEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[Test Suite] Initializing CheXNet Engine for Unit Testing...")
        cls.engine = CheXNetEngine()

    def test_01_model_loaded(self):
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(len(CLASS_NAMES), 14)
        total_params = sum(p.numel() for p in self.engine.model.parameters())
        self.assertGreater(total_params, 6_000_000)
        print(f"  [OK] Model verified: DenseNet-121 with {total_params:,} parameters.")

    def test_02_predict_patient_case(self):
        img_path = os.path.join("samples", "sample_5_nih_patient.png")
        self.assertTrue(os.path.exists(img_path))
        img = Image.open(img_path)
        
        result = self.engine.predict(img)
        self.assertIn("top_disease", result)
        self.assertIn("diseases", result)
        self.assertEqual(len(result["diseases"]), 14)
        self.assertIn("overlay", result["images"])
        self.assertIn("report", result)
        print(f"  [OK] Patient case predicted in {result['latency_ms']} ms. Top: {result['top_disease']['display']} ({result['top_disease']['percentage']}%)")

    def test_03_recompute_cam_hotspot(self):
        cam_res = self.engine.recompute_cam(1, colormap="turbo", alpha=0.5)
        self.assertIsNotNone(cam_res)
        self.assertEqual(cam_res["active_disease_name"], "Cardiomegaly")
        self.assertIn("overlay", cam_res["images"])
        print("  [OK] Real-time CAM recomputed successfully for Cardiomegaly.")

    def test_08_reject_color_photo(self):
        import numpy as np
        color_img = Image.fromarray(np.random.randint(40, 220, (300, 300, 3), dtype=np.uint8))
        res = self.engine.predict(color_img)
        self.assertFalse(res.get("is_valid_chest_xray", True))
        self.assertIn("This sample is not acceptable", res.get("rejection_note", ""))
        print(f"  [OK] Color photo properly rejected: '{res['rejection_note']}'")

    def test_09_reject_blank_image(self):
        import numpy as np
        blank_img = Image.fromarray(np.full((300, 300, 3), 128, dtype=np.uint8))
        res = self.engine.predict(blank_img)
        self.assertFalse(res.get("is_valid_chest_xray", True))
        self.assertIn("This sample is not acceptable", res.get("rejection_note", ""))
        print(f"  [OK] Blank image properly rejected: '{res['rejection_note']}'")


class TestCheXNetAPI(unittest.TestCase):
    def test_04_get_info_endpoint(self):
        r = requests.get(f"{BASE_URL}/api/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["model"]["name"], "CheXNet")
        print("  [OK] GET /api/info verified.")

    def test_05_get_samples_endpoint(self):
        r = requests.get(f"{BASE_URL}/api/samples")
        self.assertEqual(r.status_code, 200)
        samples = r.json()["samples"]
        self.assertGreaterEqual(len(samples), 4)
        print(f"  [OK] GET /api/samples returned {len(samples)} preset test cases.")

    def test_06_post_predict_endpoint(self):
        r = requests.post(
            f"{BASE_URL}/api/predict",
            data={"sample_id": "sample_cardiomegaly.png", "colormap": "jet", "alpha": "0.45"}
        )
        self.assertEqual(r.status_code, 200)
        res = r.json()
        self.assertIn("top_disease", res)
        print(f"  [OK] POST /api/predict verified for sample_cardiomegaly.png.")

    def test_07_post_benchmark_endpoint(self):
        print("  Running automated API benchmark stress test (this runs 15 real inferences)...")
        r = requests.post(f"{BASE_URL}/api/benchmark")
        self.assertEqual(r.status_code, 200)
        bench = r.json()
        self.assertEqual(bench["status"], "completed")
        print(f"  [OK] Benchmark complete: Avg Latency = {bench['average_latency_ms']} ms | Throughput = {bench['throughput_fps']} FPS.")

    def test_10_api_file_upload_rejection(self):
        import numpy as np
        import io
        # Upload non-chest image via REST API
        color_arr = np.zeros((300, 300, 3), dtype=np.uint8)
        color_arr[:, :] = [240, 120, 50] # saturated orange
        buf = io.BytesIO()
        Image.fromarray(color_arr).save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_orange.jpg", buf, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/api/predict", files=files)
        self.assertEqual(r.status_code, 200)
        res = r.json()
        self.assertFalse(res.get("is_valid_chest_xray", True))
        self.assertIn("This sample is not acceptable", res.get("rejection_note", ""))
        print(f"  [OK] API upload properly rejected non-chest sample with note: '{res['rejection_note']}'")


if __name__ == "__main__":
    unittest.main()

