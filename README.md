Cấu trúc project:
app.py — backend Flask
pipeline_web.py — phần pipeline xử lý ảnh/video, tách từ file Python bạn gửi
templates/index.html — giao diện web
static/style.css — CSS
static/app.js — xử lý upload + render kết quả
requirements.txt — thư viện cần cài
run.sh — script chạy nhanh


Vào thư mục project rồi chạy:
-- pip install -r requirements.txt
-- python app.py
