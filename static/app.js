/* ==========================================================================
   Face Recognizer ML Studio - Frontend Application Logic
   Vanilla JavaScript with Full Reactive Metric Visualizations
   ========================================================================== */

let currentEvaluationData = null;
let webcamStream = null;
let enrollStream = null;
let isRecognizing = false;
let recognitionInterval = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

// On Page Load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    triggerEvaluation();
});

// Tab Switching
function switchTab(tabName) {
    // Nav buttons
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
    const targetNav = document.getElementById(`nav-btn-${tabName}`);
    if (targetNav) targetNav.classList.add('active');

    // Tab content
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    const targetPane = document.getElementById(`tab-${tabName}-content`);
    if (targetPane) targetPane.classList.add('active');

    // Stop webcam if leaving live tab
    if (tabName !== 'live' && webcamStream) {
        toggleWebcam();
    }
    // Stop enroll camera if leaving enroll tab
    if (tabName !== 'enroll' && enrollStream) {
        startEnrollWebcam();
    }

    // Header title updates
    const titleElem = document.getElementById('header-page-title');
    const subtitleElem = document.getElementById('header-page-subtitle');
    const controlPanel = document.getElementById('evaluation-controls');

    if (tabName === 'metrics') {
        titleElem.textContent = 'KNN Model Evaluation & Metrics';
        subtitleElem.textContent = 'K-Nearest Neighbors (KNN) — Accuracy, Precision, Recall/TPR, Specificity/TNR, F1-Score, Error Rate';
        controlPanel.style.display = 'flex';
    } else if (tabName === 'live') {
        titleElem.textContent = 'Real-Time Face Recognition';
        subtitleElem.textContent = 'Webcam stream with Haar Cascade face detection & KNN inference on ./data';
        controlPanel.style.display = 'none';
    } else if (tabName === 'enroll') {
        titleElem.textContent = 'Dataset & Face Enrollment Studio';
        subtitleElem.textContent = 'Record facial training samples directly to <Name>.npy files';
        controlPanel.style.display = 'none';
        refreshDatasetSummary();
    }
}


// Hyperparameter Sliders Display
function updateKDisplay(val) {
    document.getElementById('val-k-display').textContent = val;
}

function updateSplitDisplay(val) {
    document.getElementById('val-split-display').textContent = `${val}%`;
}

// System Status Check
async function checkSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const pillText = document.getElementById('dataset-summary-text');
        if (data.has_data) {
            pillText.textContent = `./data/ (${data.local_file_count} people: ${data.registered_people.join(', ')})`;
        } else {
            pillText.textContent = 'No .npy files in ./data/ yet (Use Enroll tab)';
        }
    } catch (e) {
        console.warn('Status check:', e);
    }
}

// Run / Re-Run Evaluation Pipeline
async function triggerEvaluation() {
    const k = parseInt(document.getElementById('input-k-val').value) || 5;
    const splitPercent = parseInt(document.getElementById('input-split-val').value) || 25;
    const testSize = splitPercent / 100.0;

    const evaluateBtn = document.getElementById('btn-run-evaluation');
    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = '<span>Evaluating ./data/...</span>';

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                k: k,
                test_size: testSize
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Evaluation failed');
        }

        const data = await response.json();
        currentEvaluationData = data;
        renderAllMetrics(data);
    } catch (err) {
        alert('Evaluation Notice: ' + err.message);
    } finally {
        evaluateBtn.disabled = false;
        evaluateBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            <span>Re-Run Evaluation</span>
        `;
    }
}

// Render Metrics & Visualizations
function renderAllMetrics(data) {
    const m = data.metrics;

    // 1. Primary KPI Cards
    // Accuracy
    document.getElementById('kpi-accuracy').textContent = `${m.accuracy_percent}%`;
    document.getElementById('kpi-accuracy-sub').textContent = `${m.correct_predictions} of ${m.total_test_samples} correct test predictions`;
    document.getElementById('bar-accuracy').style.width = `${m.accuracy_percent}%`;
    const accBadge = document.getElementById('badge-accuracy-status');
    if (m.accuracy >= 0.90) {
        accBadge.textContent = 'Optimal'; accBadge.className = 'kpi-badge badge-green';
    } else if (m.accuracy >= 0.75) {
        accBadge.textContent = 'Good'; accBadge.className = 'kpi-badge badge-cyan';
    } else {
        accBadge.textContent = 'Fair'; accBadge.className = 'kpi-badge badge-amber';
    }

    // Precision
    document.getElementById('kpi-precision').textContent = m.macro_precision.toFixed(4);
    document.getElementById('bar-precision').style.width = `${Math.min(100, m.macro_precision * 100)}%`;

    // Recall / TPR
    document.getElementById('kpi-recall').textContent = m.macro_recall.toFixed(4);
    document.getElementById('bar-recall').style.width = `${Math.min(100, m.macro_recall * 100)}%`;

    // Specificity / TNR
    const specVal = (m.macro_specificity !== undefined) ? m.macro_specificity : 0;
    document.getElementById('kpi-specificity').textContent = specVal.toFixed(4);
    document.getElementById('bar-specificity').style.width = `${Math.min(100, specVal * 100)}%`;

    // F1-Score
    document.getElementById('kpi-f1').textContent = m.macro_f1.toFixed(4);
    document.getElementById('kpi-f1-sub').textContent = `Weighted F1: ${m.weighted_f1.toFixed(4)}`;
    document.getElementById('bar-f1').style.width = `${Math.min(100, m.macro_f1 * 100)}%`;

    // Error Rate
    document.getElementById('kpi-error-rate').textContent = `${m.error_rate_percent}%`;
    document.getElementById('badge-error-count').textContent = `${m.misclassifications} Error${m.misclassifications === 1 ? '' : 's'}`;
    document.getElementById('kpi-error-sub').textContent = `1 − Accuracy = ${m.error_rate_percent}%`;
    document.getElementById('bar-error').style.width = `${Math.min(100, m.error_rate_percent)}%`;

    // 2. Render Confusion Matrix Heatmap
    renderConfusionMatrix(data.confusion_matrix);

    // 3. Render K-Curve SVG Chart
    renderKCurveChart(data.k_tuning_curve, data.model_config.best_k_found);

    // 4. Render Per-Class Table
    renderClassTable(data.class_breakdown);

    // Single Class Notice Banner
    const noticeBanner = document.getElementById('single-class-banner');
    const noticeText = document.getElementById('single-class-banner-text');
    if (noticeBanner) {
        if (data.notice) {
            noticeBanner.style.display = 'flex';
            if (noticeText) noticeText.textContent = data.notice;
        } else {
            noticeBanner.style.display = 'none';
        }
    }

    // Update status text
    document.getElementById('dataset-summary-text').textContent =
        `${data.dataset_source} | ${data.dataset_info.total_samples} samples | ${data.dataset_info.num_classes} classes`;
}


// Confusion Matrix Heatmap Renderer
function renderConfusionMatrix(cmData) {
    const container = document.getElementById('matrix-container');
    const matrix = cmData.matrix;
    const labels = cmData.labels;

    if (!matrix || matrix.length === 0) {
        container.innerHTML = '<div class="loading-spinner">No matrix data available</div>';
        return;
    }

    // Find max value for color intensity
    let maxVal = 1;
    for (let r = 0; r < matrix.length; r++) {
        for (let c = 0; c < matrix[r].length; c++) {
            if (matrix[r][c] > maxVal) maxVal = matrix[r][c];
        }
    }

    let html = '<table class="cm-table"><thead><tr><th>Actual \\ Pred</th>';
    labels.forEach(lbl => {
        const shortName = lbl.length > 8 ? lbl.substring(0, 7) + '…' : lbl;
        html += `<th title="${lbl}">${shortName}</th>`;
    });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, rIdx) => {
        const rowLabel = labels[rIdx];
        const shortRowLabel = rowLabel.length > 8 ? rowLabel.substring(0, 7) + '…' : rowLabel;
        html += `<tr><td class="cm-row-label" title="${rowLabel}">${shortRowLabel}</td>`;

        row.forEach((val, cIdx) => {
            const isDiagonal = rIdx === cIdx;
            const isError = !isDiagonal && val > 0;
            let cellClass = isDiagonal ? 'cm-cell diagonal' : 'cm-cell off-diag';
            if (isDiagonal && val >= maxVal * 0.7) cellClass += ' high';
            if (isError) cellClass += ' error';

            html += `
                <td class="${cellClass}" 
                    data-actual="${labels[rIdx]}" 
                    data-pred="${labels[cIdx]}" 
                    data-count="${val}"
                    data-status="${isDiagonal ? 'True Positive' : (val > 0 ? 'Misclassification' : 'Zero Confusion')}"
                    onmouseover="showMatrixTooltip(this)"
                    onmouseout="resetMatrixTooltip()">
                    ${val}
                </td>
            `;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function showMatrixTooltip(cell) {
    const actual = cell.getAttribute('data-actual');
    const pred = cell.getAttribute('data-pred');
    const count = cell.getAttribute('data-count');
    const status = cell.getAttribute('data-status');
    const tip = document.getElementById('matrix-cell-tooltip');

    if (actual === pred) {
        tip.innerHTML = `<strong>Actual:</strong> ${actual} &rarr; <strong>Predicted:</strong> ${pred} | <span style="color: #38bdf8;"><strong>${count} Correct (${status})</strong></span>`;
    } else {
        const style = count > 0 ? 'color: #fb7185;' : 'color: #94a3b8;';
        tip.innerHTML = `<strong>Actual:</strong> ${actual} &rarr; <strong>Predicted:</strong> ${pred} | <span style="${style}"><strong>${count} Count (${status})</strong></span>`;
    }
}

function resetMatrixTooltip() {
    document.getElementById('matrix-cell-tooltip').textContent = 'Hover over any cell above to inspect exact class pair details';
}

// Interactive SVG K-Hyperparameter Chart
function renderKCurveChart(curveData, bestK) {
    const svg = document.getElementById('k-curve-svg');
    const badge = document.getElementById('optimal-k-badge');
    badge.textContent = `Optimal: k = ${bestK}`;

    if (!curveData || curveData.length === 0) {
        svg.innerHTML = '<text x="250" y="120" text-anchor="middle" fill="#94a3b8">No curve data</text>';
        return;
    }

    const width = 500;
    const height = 240;
    const padX = 50;
    const padY = 30;
    const chartW = width - padX * 2;
    const chartH = height - padY * 2;

    const minK = Math.min(...curveData.map(d => d.k));
    const maxK = Math.max(...curveData.map(d => d.k));
    const kSpan = maxK - minK || 1;

    function getX(k) {
        return padX + ((k - minK) / kSpan) * chartW;
    }

    function getYAcc(val) {
        // Val in 0..100
        return padY + chartH - (val / 100.0) * chartH;
    }

    function getYF1(f1) {
        // F1 in 0..1
        return padY + chartH - f1 * chartH;
    }

    // Build SVG paths
    let pathAcc = '';
    let pathErr = '';
    let pathF1 = '';
    let dots = '';

    curveData.forEach((d, i) => {
        const x = getX(d.k);
        const yAcc = getYAcc(d.accuracy);
        const yErr = getYAcc(d.error_rate);
        const yF1 = getYF1(d.f1_score);

        if (i === 0) {
            pathAcc += `M ${x} ${yAcc}`;
            pathErr += `M ${x} ${yErr}`;
            pathF1 += `M ${x} ${yF1}`;
        } else {
            pathAcc += ` L ${x} ${yAcc}`;
            pathErr += ` L ${x} ${yErr}`;
            pathF1 += ` L ${x} ${yF1}`;
        }

        const isOptimal = d.k === bestK;
        const star = isOptimal ? `<circle cx="${x}" cy="${yAcc}" r="7" fill="none" stroke="#38bdf8" stroke-width="2" class="pulse-circle"/>` : '';

        dots += `
            ${star}
            <circle cx="${x}" cy="${yAcc}" r="4" fill="#38bdf8">
                <title>k=${d.k} | Accuracy: ${d.accuracy}%</title>
            </circle>
            <circle cx="${x}" cy="${yErr}" r="4" fill="#fb7185">
                <title>k=${d.k} | Error Rate: ${d.error_rate}%</title>
            </circle>
            <circle cx="${x}" cy="${yF1}" r="3" fill="#818cf8">
                <title>k=${d.k} | F1: ${d.f1_score}</title>
            </circle>
            <text x="${x}" y="${height - 10}" fill="#94a3b8" font-size="11" text-anchor="middle">k=${d.k}</text>
        `;
    });

    // Grid lines
    let grid = `
        <line x1="${padX}" y1="${padY}" x2="${padX}" y2="${padY + chartH}" stroke="rgba(255,255,255,0.08)" />
        <line x1="${padX}" y1="${padY + chartH}" x2="${padX + chartW}" y2="${padY + chartH}" stroke="rgba(255,255,255,0.08)" />
        <line x1="${padX}" y1="${getYAcc(50)}" x2="${padX + chartW}" y2="${getYAcc(50)}" stroke="rgba(255,255,255,0.04)" stroke-dasharray="4" />
        <line x1="${padX}" y1="${getYAcc(100)}" x2="${padX + chartW}" y2="${getYAcc(100)}" stroke="rgba(255,255,255,0.04)" stroke-dasharray="4" />
        <text x="${padX - 8}" y="${padY + chartH + 3}" fill="#64748b" font-size="10" text-anchor="end">0%</text>
        <text x="${padX - 8}" y="${getYAcc(50) + 3}" fill="#64748b" font-size="10" text-anchor="end">50%</text>
        <text x="${padX - 8}" y="${getYAcc(100) + 3}" fill="#64748b" font-size="10" text-anchor="end">100%</text>
    `;

    svg.innerHTML = `
        ${grid}
        <path d="${pathAcc}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" />
        <path d="${pathErr}" fill="none" stroke="#fb7185" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="4 2" />
        <path d="${pathF1}" fill="none" stroke="#818cf8" stroke-width="1.8" stroke-linecap="round" />
        ${dots}
    `;
}

// Per-Class Table Renderer
function renderClassTable(classes) {
    const tbody = document.getElementById('tbody-class-metrics');
    if (!classes || classes.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="text-center">No class records available</td></tr>';
        return;
    }

    let rowsHtml = '';
    classes.forEach(c => {
        let badgeClass = 'badge-green';
        let perfText = 'Optimal';
        if (c.f1_score < 0.7) {
            badgeClass = 'badge-rose';
            perfText = 'Needs Data';
        } else if (c.f1_score < 0.9) {
            badgeClass = 'badge-amber';
            perfText = 'Good';
        }
        const spec = (c.specificity !== undefined) ? c.specificity.toFixed(4) : '--';

        rowsHtml += `
            <tr data-name="${c.name.toLowerCase()}">
                <td><strong>${c.name}</strong></td>
                <td>${c.support}</td>
                <td><span style="color: #38bdf8; font-weight: 600;">${c.tp}</span></td>
                <td><span style="color: #fb7185;">${c.fp}</span></td>
                <td>${c.fn}</td>
                <td>${c.tn}</td>
                <td>${(c.precision).toFixed(4)}</td>
                <td>${(c.recall).toFixed(4)}</td>
                <td>${spec}</td>
                <td><strong>${(c.f1_score).toFixed(4)}</strong></td>
                <td><span class="badge ${badgeClass}">${perfText}</span></td>
            </tr>
        `;
    });

    if (classes.length > 1) {
        let totalSupport = 0, totalTP = 0, totalFP = 0, totalFN = 0, totalTN = 0;
        let sumPrec = 0, sumRec = 0, sumSpec = 0, sumF1 = 0;
        classes.forEach(c => {
            totalSupport += c.support;
            totalTP += c.tp;
            totalFP += c.fp;
            totalFN += c.fn;
            totalTN += c.tn;
            sumPrec += c.precision;
            sumRec += c.recall;
            sumSpec += (c.specificity !== undefined ? c.specificity : 0);
            sumF1 += c.f1_score;
        });
        const n = classes.length;
        rowsHtml += `
            <tr style="background: rgba(99, 102, 241, 0.08); font-weight: 700; border-top: 2px solid rgba(255, 255, 255, 0.1);">
                <td><strong>Macro Average</strong></td>
                <td>${totalSupport}</td>
                <td><span style="color: #38bdf8; font-weight: 600;">${totalTP}</span></td>
                <td><span style="color: #fb7185;">${totalFP}</span></td>
                <td>${totalFN}</td>
                <td>${totalTN}</td>
                <td>${(sumPrec / n).toFixed(4)}</td>
                <td>${(sumRec / n).toFixed(4)}</td>
                <td>${(sumSpec / n).toFixed(4)}</td>
                <td><strong style="color: #a78bfa;">${(sumF1 / n).toFixed(4)}</strong></td>
                <td><span class="badge badge-accent">Average</span></td>
            </tr>
        `;
    }

    tbody.innerHTML = rowsHtml;
}

function filterClassTable(query) {
    const q = query.toLowerCase().trim();
    document.querySelectorAll('#tbody-class-metrics tr').forEach(row => {
        const name = row.getAttribute('data-name') || '';
        if (name.includes(q)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// ==========================================================================
// Live Webcam Recognition
// ==========================================================================
async function toggleWebcam() {
    const btn = document.getElementById('btn-toggle-cam');
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('camera-placeholder');

    if (webcamStream) {
        // Stop
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
        if (recognitionInterval) clearInterval(recognitionInterval);
        isRecognizing = false;
        video.srcObject = null;
        placeholder.style.display = 'flex';
        btn.textContent = 'Start Camera';
        btn.className = 'btn btn-secondary btn-sm';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });
        webcamStream = stream;
        video.srcObject = stream;
        placeholder.style.display = 'none';
        btn.textContent = 'Stop Camera';
        btn.className = 'btn btn-primary btn-sm';

        isRecognizing = true;
        recognitionInterval = setInterval(processWebcamFrame, 400); // 2.5 inferences per second
    } catch (e) {
        alert('Could not access camera: ' + e.message);
    }
}

async function processWebcamFrame() {
    if (!isRecognizing || !webcamStream) return;

    const video = document.getElementById('webcam-video');
    if (video.videoWidth === 0) return;

    const canvas = document.getElementById('hidden-frame-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

    const startTime = performance.now();
    try {
        const res = await fetch('/api/predict-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl, k: 5 })
        });
        const pred = await res.json();
        const latency = Math.round(performance.now() - startTime);

        // Update stats
        document.getElementById('stat-latency').textContent = `${latency} ms`;
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            currentFps = frameCount;
            frameCount = 0;
            lastFrameTime = now;
            document.getElementById('stat-fps').textContent = currentFps;
        }

        renderPredictionResults(pred, video);
    } catch (err) {
        console.warn('Frame predict err:', err);
    }
}

async function snapshotAndPredict() {
    if (!webcamStream) {
        await toggleWebcam();
        setTimeout(processWebcamFrame, 800);
    } else {
        processWebcamFrame();
    }
}

function renderPredictionResults(pred, video) {
    const overlay = document.getElementById('detection-overlay-canvas');
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const nameText = document.getElementById('pred-name-text');
    const confFill = document.getElementById('pred-conf-fill');
    const confText = document.getElementById('pred-conf-text');
    const avatar = document.getElementById('pred-avatar-icon');
    const neighborsGrid = document.getElementById('neighbors-grid-container');

    if (pred.detected && pred.bbox) {
        const { x, y, w, h } = pred.bbox;

        // The video is CSS-mirrored (scaleX(-1)) but the canvas is NOT.
        // We flip the X coordinate here so the bbox aligns with the mirrored display.
        // This also ensures text labels render correctly (not mirrored/backwards).
        const mx = overlay.width - x - w;

        // Draw bounding box at mirrored X position
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        ctx.strokeRect(mx, y, w, h);

        // Label banner at mirrored X position
        ctx.fillStyle = 'rgba(6, 182, 212, 0.85)';
        ctx.fillRect(mx, Math.max(0, y - 28), w, 28);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Outfit, sans-serif';
        ctx.fillText(`${pred.predicted_name} (${Math.round(pred.confidence * 100)}%)`, mx + 6, Math.max(18, y - 8));

        // Diagnostics Card
        nameText.textContent = pred.predicted_name;
        avatar.textContent = pred.predicted_name.charAt(0).toUpperCase();
        const confPct = Math.round(pred.confidence * 100);
        confFill.style.width = `${confPct}%`;
        confText.textContent = `Confidence: ${confPct}% (${pred.votes} / ${pred.k} nearest neighbors vote)`;

        // Neighbor Cards
        if (pred.neighbors && pred.neighbors.length > 0) {
            let nHtml = '';
            pred.neighbors.forEach((n, idx) => {
                nHtml += `
                    <div class="neighbor-card">
                        <div class="neighbor-info">
                            <span class="neighbor-rank">#${idx + 1}</span>
                            <span class="neighbor-name">${n.name}</span>
                        </div>
                        <span class="neighbor-dist">dist: ${n.distance}</span>
                    </div>
                `;
            });
            neighborsGrid.innerHTML = nHtml;
        }
    } else {
        nameText.textContent = 'No Face Detected';
        avatar.textContent = '?';
        confFill.style.width = '0%';
        confText.textContent = 'Looking for faces in camera view...';
        neighborsGrid.innerHTML = '<div class="neighbor-empty">Position face towards camera</div>';
    }
}

// ==========================================================================
// Face Enrollment Studio
// ==========================================================================
async function startEnrollWebcam() {
    const video = document.getElementById('enroll-video');
    const placeholder = document.getElementById('enroll-placeholder');
    const btn = document.getElementById('btn-start-enroll-cam');

    if (enrollStream) {
        enrollStream.getTracks().forEach(t => t.stop());
        enrollStream = null;
        video.srcObject = null;
        placeholder.style.display = 'flex';
        btn.textContent = 'Start Enroll Camera';
        return;
    }

    try {
        enrollStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });
        video.srcObject = enrollStream;
        placeholder.style.display = 'none';
        btn.textContent = 'Stop Enroll Camera';
    } catch (e) {
        alert('Could not start enroll camera: ' + e.message);
    }
}

// Enroll Sample Slider & Presets
function updateEnrollSamples(val) {
    const num = parseInt(val) || 100;
    const display = document.getElementById('enroll-samples-display');
    const tier = document.getElementById('enroll-samples-tier');
    const captureBtn = document.getElementById('btn-capture-faces');

    if (display) display.textContent = num;
    if (captureBtn) captureBtn.textContent = `Capture ${num} Samples`;

    if (tier) {
        if (num <= 25) {
            tier.textContent = 'Quick Test ⚡';
            tier.className = 'badge badge-amber';
        } else if (num <= 50) {
            tier.textContent = 'Balanced 👍';
            tier.className = 'badge badge-cyan';
        } else if (num <= 100) {
            tier.textContent = 'Recommended ⭐';
            tier.className = 'badge badge-accent';
        } else if (num <= 150) {
            tier.textContent = 'High Accuracy 🎯';
            tier.className = 'badge badge-green';
        } else {
            tier.textContent = 'Deep Variety 🏆';
            tier.className = 'badge badge-violet';
        }
    }

    // Update active preset chip
    document.querySelectorAll('.preset-chip').forEach(btn => {
        const chipVal = parseInt(btn.getAttribute('data-val'));
        btn.classList.toggle('active', chipVal === num);
    });
}

function setEnrollSamplesPreset(val) {
    const slider = document.getElementById('enroll-samples-slider');
    if (slider) slider.value = val;
    updateEnrollSamples(val);
}

async function captureAndEnrollFace() {
    const nameInput = document.getElementById('enroll-person-name');
    const name = nameInput.value.trim();
    if (!name) {
        alert('Please enter a person name or identity first.');
        nameInput.focus();
        return;
    }

    const slider = document.getElementById('enroll-samples-slider');
    const totalSamples = slider ? (parseInt(slider.value) || 100) : 100;
    const modeRadio = document.querySelector('input[name="enroll-mode"]:checked');
    const mode = modeRadio ? modeRadio.value : 'replace';

    if (!enrollStream) {
        await startEnrollWebcam();
        await new Promise(r => setTimeout(r, 1000));
    }

    const video = document.getElementById('enroll-video');
    const progressBox = document.getElementById('enroll-progress-box');
    const progressFill = document.getElementById('enroll-progress-fill');
    const progressText = document.getElementById('enroll-progress-text');
    const guidanceBox = document.getElementById('enroll-guidance-box');
    const guidanceText = document.getElementById('enroll-guidance-text');
    const captureBtn = document.getElementById('btn-capture-faces');

    progressBox.style.display = 'block';
    if (guidanceBox) guidanceBox.classList.add('guidance-active');
    captureBtn.disabled = true;

    // Dynamic frame delay gives user ample time to change expressions and angles
    let frameDelay = 80;
    if (totalSamples <= 25) frameDelay = 120;
    else if (totalSamples <= 50) frameDelay = 100;
    else if (totalSamples <= 100) frameDelay = 80;
    else frameDelay = 65;

    const capturedImages = [];
    const canvas = document.getElementById('hidden-frame-canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');

    for (let i = 1; i <= totalSamples; i++) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
        capturedImages.push(dataUrl);

        const pct = Math.round((i / totalSamples) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = `Capturing face samples: ${i} / ${totalSamples} (${pct}%)`;

        // Live multi-expression coaching to prevent poker-face bias
        if (guidanceText) {
            if (pct <= 25) {
                guidanceText.innerHTML = `<strong>Phase 1 (Neutral):</strong> Look straight at the camera with a neutral face (${i}/${totalSamples})`;
            } else if (pct <= 50) {
                guidanceText.innerHTML = `<strong>Phase 2 (Smile):</strong> Smile naturally, blink, or raise your eyebrows (${i}/${totalSamples})`;
            } else if (pct <= 75) {
                guidanceText.innerHTML = `<strong>Phase 3 (Turn):</strong> Slowly turn your head slightly left and right (${i}/${totalSamples})`;
            } else {
                guidanceText.innerHTML = `<strong>Phase 4 (Tilt):</strong> Tilt head slightly up and down... Almost done! (${i}/${totalSamples})`;
            }
        }

        await new Promise(r => setTimeout(r, frameDelay));
    }

    progressText.textContent = `Processing & saving ${totalSamples} face vectors to dataset...`;
    if (guidanceText) {
        guidanceText.innerHTML = `<strong>Finalizing:</strong> Running Haar face detection and saving vectors to <code>./data/${name}.npy</code>...`;
    }

    try {
        const res = await fetch('/api/dataset/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, images: capturedImages, mode: mode })
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Enrollment failed');

        alert(`Success! Enrolled ${data.name} with ${data.samples_enrolled} varied face samples (Mode: ${mode}). Retrained model!`);
        nameInput.value = '';
        progressBox.style.display = 'none';
        if (guidanceBox) guidanceBox.classList.remove('guidance-active');
        if (guidanceText) {
            guidanceText.innerHTML = `<strong>Expression Tip:</strong> While capturing, slowly change expressions — neutral, smile, slight head turns left/right/up/down. More varied samples prevent poker-face bias!`;
        }
        refreshDatasetSummary();
        triggerEvaluation();
    } catch (e) {
        alert('Enrollment error: ' + e.message);
    } finally {
        captureBtn.disabled = false;
        captureBtn.textContent = `Capture ${totalSamples} Samples`;
        if (guidanceBox) guidanceBox.classList.remove('guidance-active');
    }
}

async function refreshDatasetSummary() {
    const grid = document.getElementById('dataset-classes-grid');
    grid.innerHTML = '<div class="loading-spinner">Refreshing registered dataset...</div>';
    try {
        const res = await fetch('/api/dataset/summary');
        const data = await res.json();

        if (!data.class_distribution || Object.keys(data.class_distribution).length === 0) {
            grid.innerHTML = '<div class="neighbor-empty">No faces registered yet. Use form on left to capture or load benchmark!</div>';
            return;
        }

        let html = '';
        for (const [person, count] of Object.entries(data.class_distribution)) {
            html += `
                <div class="class-card">
                    <div class="class-avatar">${person.charAt(0).toUpperCase()}</div>
                    <div class="class-name">${person}</div>
                    <div class="class-samples">${count} face vectors</div>
                    <button class="btn-delete-identity" title="Delete ${person} from dataset" onclick="deleteIdentity('${person}')">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                        <span>Delete</span>
                    </button>
                </div>
            `;
        }
        grid.innerHTML = html;
    } catch (e) {
        grid.innerHTML = `<div class="neighbor-empty">Error loading summary: ${e.message}</div>`;
    }
}

async function deleteIdentity(name) {
    if (!confirm(`Are you sure you want to delete identity '${name}' from ./data/?`)) return;
    try {
        const res = await fetch(`/api/dataset/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Delete failed');
        await refreshDatasetSummary();
        await checkSystemStatus();
        triggerEvaluation();
    } catch (e) {
        alert('Could not delete: ' + e.message);
    }
}

