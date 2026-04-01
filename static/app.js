const form = document.getElementById('process-form');
const input = document.getElementById('media-input');
const selectedFile = document.getElementById('selected-file');
const statusBox = document.getElementById('status');
const submitBtn = document.getElementById('submit-btn');
const originalBox = document.getElementById('original-box');
const resultBox = document.getElementById('result-box');
const metricsPanel = document.getElementById('metrics-panel');
const actionRow = document.getElementById('action-row');

input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (!file) {
        selectedFile.textContent = 'Chưa chọn file';
        return;
    }
    selectedFile.textContent = `Đã chọn: ${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    renderLocalPreview(file, originalBox);
});

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = input.files?.[0];
    if (!file) {
        setStatus('Vui lòng chọn ảnh hoặc video trước khi chạy.', 'error');
        return;
    }

    submitBtn.disabled = true;
    setStatus('Đang tải file lên và xử lý. Với video dài, bạn vui lòng chờ một chút...', 'loading');
    resultBox.innerHTML = '<div class="placeholder">Đang xử lý...</div>';
    actionRow.innerHTML = '';
    metricsPanel.innerHTML = '<h3>Metrics</h3><div class="placeholder">Đang tính toán...</div>';

    const formData = new FormData(form);

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Có lỗi xảy ra trong quá trình xử lý.');
        }

        setStatus(data.message, 'success');
        renderRemotePreview(data.result_url, data.media_type, resultBox);
        renderRemotePreview(data.original_url, data.media_type, originalBox);
        renderActions(data.download_url, data.metrics_url);
        renderMetrics(data.metrics, data.config_used, data.media_type);
    } catch (error) {
        setStatus(error.message, 'error');
        resultBox.innerHTML = `<div class="placeholder">${error.message}</div>`;
    } finally {
        submitBtn.disabled = false;
    }
});

function setStatus(message, type = '') {
    statusBox.textContent = message;
    statusBox.className = `status ${type}`.trim();
}

function renderLocalPreview(file, container) {
    const type = file.type.startsWith('video/') ? 'video' : 'image';
    const url = URL.createObjectURL(file);
    renderMediaElement(url, type, container, false);
}

function renderRemotePreview(url, type, container) {
    renderMediaElement(url, type, container, true);
}

function renderMediaElement(url, type, container, withControls) {
    if (type === 'video') {
        container.innerHTML = `<video src="${url}" ${withControls ? 'controls' : 'controls muted'} playsinline></video>`;
    } else {
        container.innerHTML = `<img src="${url}" alt="preview" />`;
    }
}

function renderActions(downloadUrl, metricsUrl) {
    actionRow.innerHTML = `
        <a class="download-btn" href="${downloadUrl}">Tải output</a>
        <a class="download-btn" href="${metricsUrl}">Tải metrics JSON</a>
    `;
}

function renderMetrics(metrics, config, mediaType) {
    const items = [
        metricItem('Loại media', mediaType.toUpperCase()),
        metricItem('Tổng frames', safeValue(metrics.total_frames)),
        metricItem('FPS trung bình', safeValue(metrics.avg_fps)),
        metricItem('Seg trung bình', `${safeValue(metrics.avg_seg_ms)} ms`),
        metricItem('Det trung bình', `${safeValue(metrics.avg_det_ms)} ms`),
        metricItem('Pipeline trung bình', `${safeValue(metrics.avg_pipeline_ms)} ms`),
        metricItem('Tổng person', safeValue(metrics.total_persons_detected)),
        metricItem('Tổng on-road', safeValue(metrics.total_on_road_events)),
        metricItem('Alert frames', `${safeValue(metrics.alert_frames)} (${safeValue(metrics.alert_ratio)}%)`),
    ];

    if (metrics.input_video) {
        items.push(metricItem('Video info', `${metrics.input_video.width}x${metrics.input_video.height}`, `${metrics.input_video.fps} FPS`));
    }

    if (metrics.single_frame) {
        items.push(metricItem('Ảnh - persons', safeValue(metrics.single_frame.num_persons)));
        items.push(metricItem('Ảnh - on road', safeValue(metrics.single_frame.num_on_road)));
    }

    items.push(metricItem('Det conf', config.det_conf_threshold));
    items.push(metricItem('Overlap threshold', config.overlap_ratio_threshold));
    items.push(metricItem('Road dilate', config.road_dilate_px));

    metricsPanel.innerHTML = `
        <h3>Metrics</h3>
        <div class="metrics-grid">${items.join('')}</div>
    `;
}

function metricItem(label, value, sub = '') {
    return `
        <div class="metric-item">
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
            ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
        </div>
    `;
}

function safeValue(value) {
    return value === undefined || value === null ? '-' : value;
}
