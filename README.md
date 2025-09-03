# Dịch vụ OCR với DBNet và CRNN

Dịch vụ OCR (Optical Character Recognition) sử dụng mô hình *DBNet* để phát hiện văn bản và *CRNN* để nhận dạng văn bản.  
Dự án đã được đóng gói sẵn trong *Docker*, dễ dàng triển khai và chạy trên môi trường có hỗ trợ *CUDA*.

---

## :sparkles: Tính năng chính
- *Phát hiện văn bản:* Sử dụng mô hình *DBNet*
- *Nhận dạng văn bản:* Sử dụng mô hình *CRNN*
- *Triển khai nhanh chóng:* Qua *Docker Compose*
- *Hỗ trợ GPU:* Tận dụng *CUDA* để tăng tốc độ xử lý

---

## :gear: Yêu cầu hệ thống
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)
- GPU với hỗ trợ *CUDA* (khuyến nghị để tăng hiệu suất)

---

## :rocket: Hướng dẫn cài đặt và chạy

### 1. Build docker images (current use cuda image to base env)
```
sudo docker compose build
```

### 2. Run
```
sudo docker compose up -d
```
default the ocr service run on port 8312
