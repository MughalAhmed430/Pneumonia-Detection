/* ==========================================================================
   CheXNet Medical AI Diagnostic Platform — Client Logic (app.js)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentAnalysis = null;
  let activeDiseaseIdx = null;
  let activeViewMode = 'split'; // 'split' | 'overlay' | 'side-by-side'
  let showHotspotBox = true;
  let isInverted = false;
  let isDraggingSplit = false;
  let splitPercent = 50;
  let activeSampleId = null;

  // DOM Elements
  const stagePlaceholder = document.getElementById('stage-placeholder');
  const loadingOverlay = document.getElementById('loading-overlay');
  const splitContainer = document.getElementById('split-container');
  const splitOverlayWrapper = document.getElementById('split-overlay-wrapper');
  const splitHandle = document.getElementById('split-handle');
  const sideBySideContainer = document.getElementById('side-by-side-container');
  
  const imgBase = document.getElementById('img-base');
  const imgCamOverlay = document.getElementById('img-cam-overlay');
  const imgSideBase = document.getElementById('img-side-base');
  const imgSideCam = document.getElementById('img-side-cam');

  const presetList = document.getElementById('preset-list');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  const diseaseGrid = document.getElementById('disease-grid');
  const reportContent = document.getElementById('report-content');
  const btnCopyReport = document.getElementById('btn-copy-report');

  const alphaSlider = document.getElementById('alpha-slider');
  const alphaVal = document.getElementById('alpha-val');
  const colormapSelect = document.getElementById('colormap-select');
  const btnToggleBox = document.getElementById('btn-toggle-box');
  const btnInvert = document.getElementById('btn-invert');

  const activeBanner = document.getElementById('active-pathology-banner');
  const activePathologyName = document.getElementById('active-pathology-name');
  const activePathologyProb = document.getElementById('active-pathology-prob');

  const metaPrimary = document.getElementById('meta-primary');
  const metaLatency = document.getElementById('meta-latency');
  const metaHotspot = document.getElementById('meta-hotspot');
  const metaResolution = document.getElementById('meta-resolution');

  // Rejection Overlay Elements
  const rejectionOverlay = document.getElementById('rejection-overlay');
  const rejectionNoteText = document.getElementById('rejection-note-text');
  const btnRejectionPreset = document.getElementById('btn-rejection-preset');
  const btnRejectionDismiss = document.getElementById('btn-rejection-dismiss');

  // Modals
  const benchmarkModal = document.getElementById('benchmark-modal');
  const btnBenchmarkModal = document.getElementById('btn-benchmark-modal');
  const btnCloseBenchmark = document.getElementById('btn-close-benchmark');
  const btnCloseBenchmarkFooter = document.getElementById('btn-close-benchmark-footer');
  const btnRunBenchmark = document.getElementById('btn-run-benchmark');
  const benchStatus = document.getElementById('bench-status');
  const benchTrialsTable = document.getElementById('bench-trials-table');

  const infoModal = document.getElementById('info-modal');
  const btnInfoModal = document.getElementById('btn-info-modal');
  const btnCloseInfo = document.getElementById('btn-close-info');
  const btnCloseInfoFooter = document.getElementById('btn-close-info-footer');

  // ==========================================================================
  // 1. Initial Load & Setup
  // ==========================================================================
  initSystem();

  async function initSystem() {
    setupSplitWipe();
    setupViewModeTabs();
    setupDragDrop();
    setupControls();
    setupModals();

    await loadModelInfo();
    await loadSamples();
  }

  // ==========================================================================
  // 2. Fetch Info & Samples
  // ==========================================================================
  async function loadModelInfo() {
    try {
      const res = await fetch('/api/info');
      const data = await res.json();
      if (data.status === 'online') {
        document.getElementById('model-status-text').textContent = `DenseNet-121 (${data.model.device.toUpperCase()})`;
        document.getElementById('bench-device').textContent = data.model.device.toUpperCase();
        document.getElementById('bench-params').textContent = (data.model.parameters / 1e6).toFixed(2) + 'M';
        
        // Populate AUROC table in benchmark modal
        const aurocTbody = document.getElementById('auroc-tbody');
        if (aurocTbody && data.benchmarks) {
          aurocTbody.innerHTML = data.benchmarks.map(b => {
            const delta = ((b.our_impl - b.chexnet) * 100).toFixed(2);
            const deltaSign = delta >= 0 ? `+${delta}%` : `${delta}%`;
            return `
              <tr>
                <td style="font-family: var(--font-main); font-weight: 600;">${b.disease}</td>
                <td>${b.chexnet.toFixed(4)}</td>
                <td style="color: var(--primary-cyan); font-weight: 700;">${b.our_impl.toFixed(4)}</td>
                <td>${b.improved.toFixed(4)}</td>
                <td class="${delta >= 0 ? 'delta-pos' : ''}">${deltaSign}</td>
              </tr>
            `;
          }).join('');
        }
      }
    } catch (err) {
      console.error('Failed to load model info:', err);
    }
  }

  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        renderPresetList(data.samples);
        // Auto-run the first sample for immediate visualization
        analyzePreset(data.samples[0].id);
      } else {
        presetList.innerHTML = `<div class="preset-loading">No sample radiographs found.</div>`;
      }
    } catch (err) {
      console.error('Failed to load samples:', err);
      presetList.innerHTML = `<div class="preset-loading">Error loading samples.</div>`;
    }
  }

  function renderPresetList(samples) {
    presetList.innerHTML = samples.map((sample, idx) => `
      <div class="preset-card ${idx === 0 ? 'active' : ''}" data-id="${sample.id}">
        <img src="${sample.url}" alt="${sample.title}" class="preset-thumb">
        <div class="preset-info">
          <div class="preset-title">${sample.title}</div>
          <div class="preset-tag">${sample.category}</div>
          <div class="preset-desc">${sample.description}</div>
        </div>
      </div>
    `).join('');

    // Attach click handlers
    document.querySelectorAll('.preset-card').forEach(card => {
      card.addEventListener('click', () => {
        const sampleId = card.getAttribute('data-id');
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        activeSampleId = sampleId;
        analyzePreset(sampleId);
      });
    });
  }

  // ==========================================================================
  // 3. Inference & Prediction Execution
  // ==========================================================================
  async function analyzePreset(sampleId) {
    activeSampleId = sampleId;
    const formData = new FormData();
    formData.append('sample_id', sampleId);
    formData.append('colormap', colormapSelect.value);
    formData.append('alpha', alphaSlider.value);
    await executePrediction(formData);
  }

  async function analyzeUploadedFile(file) {
    activeSampleId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('colormap', colormapSelect.value);
    formData.append('alpha', alphaSlider.value);
    await executePrediction(formData);
  }

  async function executePrediction(formData) {
    showLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await res.json();
      currentAnalysis = data;
      activeDiseaseIdx = data.active_disease_idx;
      
      renderAnalysis(data);
    } catch (err) {
      alert('Analysis Error: ' + err.message);
      console.error(err);
    } finally {
      showLoading(false);
    }
  }

  // ==========================================================================
  // 4. Render Analysis in Viewer, Matrix & Report
  // ==========================================================================
  function renderAnalysis(data) {
    stagePlaceholder.style.display = 'none';

    // Handle Rejected Non-Chest Samples
    if (data.is_valid_chest_xray === false) {
      rejectionOverlay.style.display = 'flex';
      rejectionNoteText.textContent = data.rejection_note || 'This sample is not acceptable. Only frontal chest (lung) radiographs are accepted by this model.';

      if (data.images && data.images.original) {
        imgBase.src = data.images.original;
        imgCamOverlay.src = data.images.original;
        imgSideBase.src = data.images.original;
        imgSideCam.src = data.images.original;
      }

      metaPrimary.textContent = 'SAMPLE REJECTED';
      metaPrimary.style.color = 'var(--risk-high)';
      metaLatency.textContent = 'Validation Failed';
      metaLatency.style.color = 'var(--risk-high)';
      metaHotspot.textContent = 'Non-Chest Image';
      metaResolution.textContent = '—';

      activeBanner.style.display = 'none';

      diseaseGrid.innerHTML = `
        <div class="grid-empty" style="color: var(--risk-high); padding: 3rem 1.5rem; text-align: center;">
          <div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.5rem; color: #fff;">
            ⚠️ Thoracic Diagnosis Suspended
          </div>
          <div style="font-size: 0.9rem; color: #fda4af; max-width: 580px; margin: 0 auto 0.75rem;">
            ${data.rejection_note}
          </div>
          <div style="font-size: 0.78rem; color: var(--text-dim); max-width: 540px; margin: 0 auto;">
            CheXNet is strictly tuned to frontal chest radiographs (PA/AP view). It rejects non-chest inputs to prevent false-positive diagnostic conclusions.
          </div>
        </div>
      `;

      reportContent.innerHTML = `
        <div style="padding: 0.85rem; border-left: 3px solid var(--risk-high); background: rgba(244, 63, 94, 0.08); border-radius: var(--radius-sm);">
          <div style="color: var(--risk-high); font-weight: 700; font-size: 0.78rem; margin-bottom: 0.35rem; text-transform: uppercase;">
            DIAGNOSTIC STATUS: SAMPLE REJECTED
          </div>
          <div style="color: var(--text-main); font-size: 0.82rem; line-height: 1.5; font-weight: 600;">
            ${data.rejection_note}
          </div>
          <div style="color: var(--text-dim); font-size: 0.72rem; margin-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.4rem;">
            Requirement: Frontal view posteroanterior (PA) or anteroposterior (AP) thoracic radiograph of adult pulmonary fields.
          </div>
        </div>
      `;

      return;
    }

    // Valid chest X-ray: hide rejection banner and restore styles
    rejectionOverlay.style.display = 'none';
    metaPrimary.style.color = 'var(--primary-cyan)';
    metaLatency.style.color = 'var(--primary-cyan)';

    // Set images
    imgBase.src = data.images.original;
    imgSideBase.src = data.images.original;

    const overlayImgSrc = showHotspotBox ? data.images.overlay_with_box : data.images.overlay;
    imgCamOverlay.src = overlayImgSrc;
    imgSideCam.src = overlayImgSrc;

    // Set Meta bar
    metaPrimary.textContent = `${data.top_disease.display} (${data.top_disease.percentage}%)`;
    metaLatency.textContent = `${data.latency_ms} ms`;
    metaResolution.textContent = `${data.image_dims.width} × ${data.image_dims.height}`;
    
    if (data.hotspot) {
      metaHotspot.textContent = `[X:${data.hotspot.x_pct}%, Y:${data.hotspot.y_pct}%, W:${data.hotspot.w_pct}%]`;
    } else {
      metaHotspot.textContent = 'Diffuse / None';
    }

    // Active CAM Focus Banner
    activeBanner.style.display = 'flex';
    activePathologyName.textContent = data.active_disease_name;
    const activeProbObj = data.diseases.find(d => d.index === activeDiseaseIdx);
    activePathologyProb.textContent = `${activeProbObj ? activeProbObj.percentage : 0}% Probability`;

    // Render 14 Disease Breakdown Grid
    renderDiseaseMatrix(data.diseases, activeDiseaseIdx);

    // Render Clinical Memo
    renderClinicalReport(data.report);
  }

  function renderDiseaseMatrix(diseases, activeIdx) {
    diseaseGrid.innerHTML = diseases.map(disease => {
      let severityClass = 'severity-low';
      let riskLabel = 'Low / Normal';
      if (disease.probability >= 0.50) {
        severityClass = 'severity-high';
        riskLabel = 'High Risk';
      } else if (disease.probability >= 0.20) {
        severityClass = 'severity-moderate';
        riskLabel = 'Moderate';
      }

      const isActive = disease.index === activeIdx;

      return `
        <div class="disease-card ${severityClass} ${isActive ? 'active' : ''}" data-idx="${disease.index}">
          <div class="disease-card-header">
            <div class="disease-title-row">
              <span class="disease-name">${disease.display}</span>
              <span class="disease-category">${disease.category}</span>
            </div>
            <div class="disease-prob-text">${disease.percentage}%</div>
          </div>
          
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${disease.percentage}%;"></div>
          </div>

          <div class="disease-card-footer">
            <span class="disease-risk-tag">${riskLabel}</span>
            <span class="cam-badge">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <polygon points="10 8 16 12 10 16 10 8"></polygon>
              </svg>
              ${isActive ? 'Active CAM' : 'Project CAM'}
            </span>
          </div>
        </div>
      `;
    }).join('');

    // Attach click handlers to disease cards for instant CAM projection
    document.querySelectorAll('.disease-card').forEach(card => {
      card.addEventListener('click', () => {
        const idx = parseInt(card.getAttribute('data-idx'), 10);
        switchDiseaseCam(idx);
      });
    });
  }

  // ==========================================================================
  // 5. Fast Real-Time CAM Switching (<10ms)
  // ==========================================================================
  async function switchDiseaseCam(diseaseIdx) {
    if (!currentAnalysis) return;
    activeDiseaseIdx = diseaseIdx;

    // Highlight card
    document.querySelectorAll('.disease-card').forEach(card => {
      const idx = parseInt(card.getAttribute('data-idx'), 10);
      if (idx === diseaseIdx) {
        card.classList.add('active');
        const badge = card.querySelector('.cam-badge');
        if (badge) badge.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polygon points="10 8 16 12 10 16 10 8"></polygon>
          </svg>
          Active CAM
        `;
      } else {
        card.classList.remove('active');
        const badge = card.querySelector('.cam-badge');
        if (badge) badge.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polygon points="10 8 16 12 10 16 10 8"></polygon>
          </svg>
          Project CAM
        `;
      }
    });

    try {
      const res = await fetch('/api/recompute_cam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          disease_idx: diseaseIdx,
          colormap: colormapSelect.value,
          alpha: parseFloat(alphaSlider.value)
        })
      });

      if (!res.ok) throw new Error('CAM recompute failed');
      const camData = await res.json();

      const overlaySrc = showHotspotBox ? camData.images.overlay_with_box : camData.images.overlay;
      imgCamOverlay.src = overlaySrc;
      imgSideCam.src = overlaySrc;

      // Update banner
      activePathologyName.textContent = camData.active_disease_name;
      const diseaseObj = currentAnalysis.diseases.find(d => d.index === diseaseIdx);
      activePathologyProb.textContent = `${diseaseObj ? diseaseObj.percentage : 0}% Probability`;

      if (camData.hotspot) {
        metaHotspot.textContent = `[X:${camData.hotspot.x_pct}%, Y:${camData.hotspot.y_pct}%, W:${camData.hotspot.w_pct}%]`;
      } else {
        metaHotspot.textContent = 'Diffuse / None';
      }
    } catch (err) {
      console.error('Error switching CAM:', err);
    }
  }

  // ==========================================================================
  // 6. Clinical Radiology Report
  // ==========================================================================
  function renderClinicalReport(rep) {
    if (!rep) return;
    reportContent.innerHTML = `
      <div class="report-meta-grid">
        <div>
          <div class="report-item-label">Study Modality</div>
          <div class="report-item-val">${rep.study_type}</div>
        </div>
        <div>
          <div class="report-item-label">Input Matrix</div>
          <div class="report-item-val">${rep.resolution}</div>
        </div>
        <div>
          <div class="report-item-label">AI Inference Time</div>
          <div class="report-item-val">${rep.analysis_time}</div>
        </div>
        <div>
          <div class="report-item-label">Primary Finding</div>
          <div class="report-item-val" style="color: var(--primary-cyan);">${rep.primary_finding}</div>
        </div>
      </div>

      <div class="report-section-title">Radiological Findings & Impression</div>
      <div class="report-text">${rep.clinical_summary}</div>

      <div class="report-section-title">Clinical Recommendation</div>
      <div class="report-text" style="color: var(--text-dim); font-size: 0.72rem;">${rep.recommendation}</div>
    `;
  }

  btnCopyReport.addEventListener('click', () => {
    if (!currentAnalysis || !currentAnalysis.report) return;
    const rep = currentAnalysis.report;
    const textToCopy = `CHEXNET THORACIC RADIOLOGY REPORT
Study: ${rep.study_type} (${rep.resolution})
Inference Latency: ${rep.analysis_time}
Primary Finding: ${rep.primary_finding}

FINDINGS & IMPRESSION:
${rep.clinical_summary}

RECOMMENDATION:
${rep.recommendation}
Generated by CheXNet AI Diagnostic System.`;

    navigator.clipboard.writeText(textToCopy).then(() => {
      const origHtml = btnCopyReport.innerHTML;
      btnCopyReport.innerHTML = `✓ Copied`;
      setTimeout(() => { btnCopyReport.innerHTML = origHtml; }, 1800);
    });
  });

  // ==========================================================================
  // 7. Interactive Wipe-Slider Comparison
  // ==========================================================================
  function setupSplitWipe() {
    const updateSplitPosition = (clientX) => {
      const rect = splitContainer.getBoundingClientRect();
      let pos = ((clientX - rect.left) / rect.width) * 100;
      pos = Math.max(0, Math.min(100, pos));
      splitPercent = pos;

      splitOverlayWrapper.style.width = `${pos}%`;
      splitHandle.style.left = `${pos}%`;
    };

    // Mouse events
    splitHandle.addEventListener('mousedown', (e) => {
      isDraggingSplit = true;
      splitHandle.classList.add('dragging');
      e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
      if (isDraggingSplit) {
        isDraggingSplit = false;
        splitHandle.classList.remove('dragging');
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingSplit) return;
      updateSplitPosition(e.clientX);
    });

    // Touch events for mobile/tablet
    splitHandle.addEventListener('touchstart', () => {
      isDraggingSplit = true;
      splitHandle.classList.add('dragging');
    });

    window.addEventListener('touchend', () => {
      isDraggingSplit = false;
      splitHandle.classList.remove('dragging');
    });

    window.addEventListener('touchmove', (e) => {
      if (!isDraggingSplit || !e.touches[0]) return;
      updateSplitPosition(e.touches[0].clientX);
    });
  }

  // ==========================================================================
  // 8. View Mode Tabs (Split Wipe, Blended Overlay, Side-by-Side)
  // ==========================================================================
  function setupViewModeTabs() {
    const tabs = document.querySelectorAll('.mode-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        activeViewMode = tab.getAttribute('data-mode');

        if (activeViewMode === 'split') {
          splitContainer.style.display = 'block';
          sideBySideContainer.style.display = 'none';
          splitOverlayWrapper.style.width = `${splitPercent}%`;
          splitHandle.style.display = 'flex';
          document.getElementById('opacity-tool-group').style.display = 'flex';
        } else if (activeViewMode === 'overlay') {
          splitContainer.style.display = 'block';
          sideBySideContainer.style.display = 'none';
          splitOverlayWrapper.style.width = '100%';
          splitHandle.style.display = 'none';
          document.getElementById('opacity-tool-group').style.display = 'flex';
        } else if (activeViewMode === 'side-by-side') {
          splitContainer.style.display = 'none';
          sideBySideContainer.style.display = 'grid';
          document.getElementById('opacity-tool-group').style.display = 'flex';
        }
      });
    });
  }

  // ==========================================================================
  // 9. Viewer Controls (Opacity, Colormap, Box, Invert)
  // ==========================================================================
  function setupControls() {
    // Opacity slider
    alphaSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      alphaVal.textContent = `${Math.round(val * 100)}%`;
      if (activeDiseaseIdx !== null) {
        switchDiseaseCam(activeDiseaseIdx);
      }
    });

    // Colormap selector
    colormapSelect.addEventListener('change', () => {
      if (activeDiseaseIdx !== null) {
        switchDiseaseCam(activeDiseaseIdx);
      }
    });

    // Hotspot bounding box toggle
    btnToggleBox.addEventListener('click', () => {
      showHotspotBox = !showHotspotBox;
      btnToggleBox.classList.toggle('active', showHotspotBox);
      if (activeDiseaseIdx !== null) {
        switchDiseaseCam(activeDiseaseIdx);
      }
    });

    // Invert Grayscale Radiograph
    btnInvert.addEventListener('click', () => {
      isInverted = !isInverted;
      imgBase.classList.toggle('inverted', isInverted);
      imgSideBase.classList.toggle('inverted', isInverted);
      btnInvert.classList.toggle('btn-primary', isInverted);
      btnInvert.classList.toggle('btn-outline', !isInverted);
    });
  }

  // ==========================================================================
  // 10. File Drag & Drop Upload
  // ==========================================================================
  function setupDragDrop() {
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        analyzeUploadedFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        analyzeUploadedFile(e.dataTransfer.files[0]);
      }
    });
  }

  // ==========================================================================
  // 11. Modals & Benchmark Stress Test Execution
  // ==========================================================================
  function setupModals() {
    // Benchmark modal
    btnBenchmarkModal.addEventListener('click', () => { benchmarkModal.style.display = 'flex'; });
    btnCloseBenchmark.addEventListener('click', () => { benchmarkModal.style.display = 'none'; });
    btnCloseBenchmarkFooter.addEventListener('click', () => { benchmarkModal.style.display = 'none'; });

    // Info modal
    btnInfoModal.addEventListener('click', () => { infoModal.style.display = 'flex'; });
    btnCloseInfo.addEventListener('click', () => { infoModal.style.display = 'none'; });
    btnCloseInfoFooter.addEventListener('click', () => { infoModal.style.display = 'none'; });

    // Close on backdrop click
    window.addEventListener('click', (e) => {
      if (e.target === benchmarkModal) benchmarkModal.style.display = 'none';
      if (e.target === infoModal) infoModal.style.display = 'none';
    });

    // Rejection buttons
    if (btnRejectionDismiss) {
      btnRejectionDismiss.addEventListener('click', () => {
        rejectionOverlay.style.display = 'none';
      });
    }

    if (btnRejectionPreset) {
      btnRejectionPreset.addEventListener('click', () => {
        rejectionOverlay.style.display = 'none';
        const firstCard = document.querySelector('.preset-card');
        if (firstCard) {
          firstCard.click();
        } else {
          analyzePreset('sample_5_nih_patient.png');
        }
      });
    }

    // Execute Benchmark Stress Test
    btnRunBenchmark.addEventListener('click', async () => {
      btnRunBenchmark.disabled = true;
      btnRunBenchmark.innerHTML = `
        <div class="spinner-ring" style="width:16px;height:16px;margin:0;border-width:2px;display:inline-block;vertical-align:middle;"></div>
        Benchmarking (15 Real Inferences)...
      `;
      benchStatus.textContent = 'Executing multi-batch stress test across sample images...';

      try {
        const res = await fetch('/api/benchmark', { method: 'POST' });
        const benchData = await res.json();

        if (!res.ok) throw new Error(benchData.error || 'Benchmark failed');

        document.getElementById('bench-avg-lat').textContent = `${benchData.average_latency_ms} ms`;
        document.getElementById('bench-fps').textContent = `${benchData.throughput_fps} FPS`;
        benchStatus.textContent = `Completed ${benchData.total_inferences} inferences in ${benchData.total_time_seconds}s!`;

        // Render trials table
        const tbody = benchTrialsTable.querySelector('tbody');
        tbody.innerHTML = benchData.trials.map(t => `
          <tr>
            <td style="font-family: var(--font-main); font-weight: 600;">${t.sample}</td>
            <td style="color: var(--primary-cyan); font-weight: 700;">${t.top_prediction}</td>
            <td>${t.top_confidence}</td>
            <td>${t.min_ms}</td>
            <td style="color: #fff; font-weight: 700;">${t.avg_ms}</td>
            <td>${t.max_ms}</td>
          </tr>
        `).join('');

      } catch (err) {
        benchStatus.textContent = 'Error: ' + err.message;
      } finally {
        btnRunBenchmark.disabled = false;
        btnRunBenchmark.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          Re-Execute Stress Test
        `;
      }
    });
  }

  function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }
});
