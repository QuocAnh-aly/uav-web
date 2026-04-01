import json
import os
import shutil
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pipeline_web import DEFAULT_CONFIG, get_pipeline_cached, process_image_file, process_video_file


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULT_DIR = BASE_DIR / "data" / "results"
METRICS_DIR = BASE_DIR / "data" / "metrics"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

for folder in [UPLOAD_DIR, RESULT_DIR, METRICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB


def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS


def media_type_from_extension(filename):
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return None


def build_config_from_request(form):
    return {
        **DEFAULT_CONFIG,
        "road_class_id": int(form.get("road_class_id", DEFAULT_CONFIG["road_class_id"])),
        "person_seg_class_id": int(form.get("person_seg_class_id", DEFAULT_CONFIG["person_seg_class_id"])),
        "det_conf_threshold": float(form.get("det_conf_threshold", DEFAULT_CONFIG["det_conf_threshold"])),
        "det_iou_threshold": float(form.get("det_iou_threshold", DEFAULT_CONFIG["det_iou_threshold"])),
        "overlap_ratio_threshold": float(form.get("overlap_ratio_threshold", DEFAULT_CONFIG["overlap_ratio_threshold"])),
        "road_dilate_px": int(form.get("road_dilate_px", DEFAULT_CONFIG["road_dilate_px"])),
        "det_input_size": int(form.get("det_input_size", DEFAULT_CONFIG["det_input_size"])),
        # TTA config
        "use_tta": form.get("use_tta", "true").lower() in ("true", "1", "on", "yes"),
        "tta_flip_h": form.get("tta_flip_h", "true").lower() in ("true", "1", "on", "yes"),
        "tta_flip_v": form.get("tta_flip_v", "true").lower() in ("true", "1", "on", "yes"),
        "tta_scales": [1.0],
    }


@app.route("/")
def index():
    return render_template("index.html", defaults=DEFAULT_CONFIG)


@app.route("/api/process", methods=["POST"])
def process_media():
    if "media" not in request.files:
        return jsonify({"error": "Thiếu file media."}), 400

    media = request.files["media"]
    if not media or not media.filename:
        return jsonify({"error": "Bạn chưa chọn file."}), 400

    if not allowed_file(media.filename):
        return jsonify({"error": "Định dạng file chưa được hỗ trợ."}), 400

    seg_model = request.form.get("seg_model", "seg/modelseg.onnx").strip()
    det_model = request.form.get("det_model", "detection.onnx").strip()

    if not os.path.exists(seg_model):
        return jsonify({"error": f"Không tìm thấy segmentation model: {seg_model}"}), 400
    if not os.path.exists(det_model):
        return jsonify({"error": f"Không tìm thấy detection model: {det_model}"}), 400

    file_id = uuid.uuid4().hex
    original_name = secure_filename(media.filename)
    ext = Path(original_name).suffix.lower()
    input_filename = f"{file_id}{ext}"
    input_path = UPLOAD_DIR / input_filename
    media.save(input_path)

    config = build_config_from_request(request.form)
    config_signature = json.dumps(config, sort_keys=True)
    pipeline = get_pipeline_cached(seg_model, det_model, config_signature)

    mtype = media_type_from_extension(original_name)
    if not mtype:
        return jsonify({"error": "Không xác định được loại media."}), 400

    try:
        if mtype == "image":
            output_filename = f"{file_id}_result{ext if ext in {'.png', '.jpg', '.jpeg', '.webp'} else '.jpg'}"
            output_path = RESULT_DIR / output_filename
            summary = process_image_file(pipeline, str(input_path), str(output_path))
            metrics_filename = f"{file_id}_metrics.json"
            metrics_path = METRICS_DIR / metrics_filename
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        else:
            output_filename = f"{file_id}_result.mp4"
            output_path = RESULT_DIR / output_filename
            summary, generated_metrics_path = process_video_file(pipeline, str(input_path), str(output_path))
            metrics_filename = f"{file_id}_metrics.json"
            metrics_path = METRICS_DIR / metrics_filename
            shutil.copyfile(generated_metrics_path, metrics_path)

        return jsonify({
            "message": "Xử lý thành công.",
            "media_type": mtype,
            "original_name": original_name,
            "original_url": url_for("serve_upload", filename=input_filename),
            "result_url": url_for("serve_result", filename=output_filename),
            "download_url": url_for("download_result", filename=output_filename),
            "metrics_url": url_for("download_metrics", filename=metrics_path.name),
            "metrics": summary,
            "config_used": config,
        })
    except Exception as exc:
        return jsonify({"error": f"Xử lý thất bại: {exc}"}), 500


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/download/<path:filename>")
def download_result(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


@app.route("/metrics/<path:filename>")
def download_metrics(filename):
    return send_from_directory(METRICS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

