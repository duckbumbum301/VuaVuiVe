# SETUP DỰ ÁN - HƯỚNG DẪN NHANH

## Yêu cầu

- Node.js >= 16.0.0
- npm
- Port 3000, 8000, 8888 trống

## 3 BƯỚC SETUP

### 1️ Clone dự án

```bash
git clone https://github.com/duckbumbum301/Group5_FinalProject.git
cd Group5_FinalProject
```

### 2️ Cài đặt

```bash
.\setup.bat
```

### 3️ Chạy

```bash
.\start-all.bat
```

## Truy cập

- **Trang chủ**: http://localhost:8000/html/index.html
- **Admin**: http://localhost:8000/backoffice/
- **Test VNPay**: http://localhost:8000/test-vnpay-flow.html

## Dừng

```bash
.\stop-all.bat
```

## 🔧 Lỗi thường gặp

**Port bị chiếm:**

```bash
taskkill /F /IM node.exe
```

**Thiếu module:**

```bash
npm install
cd vnpay_nodejs
npm install
```

**Script không chạy:**

```bash
powershell -ExecutionPolicy Bypass .\start-all.bat
```

---

**Xem chi tiết:** `doc/SETUP_GUIDE.md`
