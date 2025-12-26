#  Hướng dẫn Setup Project - Vựa Vui Vẻ

> **Dành cho thành viên mới** - Setup project trong 10 phút

---

##  Yêu cầu hệ thống

Trước khi bắt đầu, đảm bảo máy tính của bạn đã cài đặt:

-  **Node.js** (phiên bản >= 16.0.0) - [Download tại đây](https://nodejs.org/)
-  **Git** - [Download tại đây](https://git-scm.com/)
-  **Trình duyệt web** (Chrome, Firefox, Edge...)
-  **Code editor** (VS Code khuyến nghị)

### Kiểm tra cài đặt

Mở terminal (Command Prompt, PowerShell hoặc Git Bash) và chạy:

```bash
node --version
# Kết quả mong đợi: v16.x.x hoặc cao hơn

npm --version
# Kết quả mong đợi: 8.x.x hoặc cao hơn

git --version
# Kết quả mong đợi: git version 2.x.x
```

---

##  Bước 1: Clone Project

### Option 1: Clone qua HTTPS

```bash
# Mở terminal tại thư mục bạn muốn lưu project
cd Desktop

# Clone repository
git clone https://github.com/duckbumbum301/Group5_FinalProject.git

# Di chuyển vào thư mục project
cd Group5_FinalProject/Group5_FinalProject
```

### Option 2: Clone qua SSH (Nếu đã setup SSH key)

```bash
git clone git@github.com:duckbumbum301/Group5_FinalProject.git
cd Group5_FinalProject/Group5_FinalProject
```

### Option 3: Download ZIP

1. Vào https://github.com/duckbumbum301/Group5_FinalProject
2. Click nút **Code** → **Download ZIP**
3. Giải nén file ZIP
4. Mở terminal tại thư mục đã giải nén

---

##  Bước 2: Cài đặt Dependencies

```bash
# Cài đặt tất cả packages cần thiết
npm install
```

**Chờ khoảng 1-2 phút...**

Kết quả mong đợi:

```
added 150 packages, and audited 151 packages in 45s
```

###  Nếu gặp lỗi:

#### Lỗi: "npm not found"

```bash
# Tải và cài Node.js từ https://nodejs.org/
# Sau đó restart terminal và thử lại
```

#### Lỗi: "EACCES" hoặc "permission denied"

```bash
# Windows: Chạy terminal với quyền Administrator
# Mac/Linux: Thêm sudo trước lệnh
sudo npm install
```

#### Lỗi: "ECONNREFUSED" hoặc network error

```bash
# Kiểm tra kết nối internet
# Hoặc thử với VPN nếu bị chặn
```

---

##  Bước 3: Test hệ thống (Optional nhưng khuyến nghị)

```bash
npm run test:api
```

Kết quả mong đợi:

```
 Bắt đầu test Data Manager API...

 Testing Products API...
 Get all products: 86 sản phẩm
 Get product by ID: Rau muống (500g)
 Filter by category 'veg': 38 sản phẩm

 Testing Orders API...
 Get all orders: 4 đơn hàng

 Tất cả tests hoàn thành!
```

---

##  Bước 4: Chạy Project

### Cách 1: Chạy tất cả (Backend + Frontend) - KHUYẾN NGHỊ 

```bash
npm start
```

Kết quả:

```
 Vựa Vui Vẻ API Server đang chạy!
 URL: http://localhost:3000
...
Starting up http-server, serving ./
Available on:
  http://localhost:8000
```
##  Bước 5: Truy cập Website

---

##  Scripts NPM có sẵn

```bash
# Chạy cả backend + frontend
npm start

# Chỉ chạy backend (Express API Server)
npm run backend

# Chỉ chạy frontend (Static file server)
npm run frontend

# Test API
npm run test:api

# Đồng bộ products từ data.js
npm run sync
```


##  Workflow làm việc

### 1. Lần đầu setup (1 lần duy nhất)

```bash
git clone <repo>
cd Group5_FinalProject/Group5_FinalProject
npm install
npm run test:api
```

### 2. Mỗi ngày làm việc

```bash
# Pull code mới nhất
git pull origin main

# Cài đặt dependencies mới (nếu có)
npm install

# Chạy project
npm start

# Làm việc...

# Commit & push
git add .
git commit -m "Your message"
git push origin main
```

### 3. Test trước khi commit

```bash
# Test API
npm run test:api

# Test manually trên browser
# - Xem danh sách sản phẩm
# - Thêm vào giỏ hàng
# - Checkout
# - Backoffice: CRUD operations
```

---

##  Nhiệm vụ đầu tiên

Để làm quen với project, thử các tác vụ sau:

### 1️ Xem sản phẩm trên frontend

- Vào http://localhost:8000
- Browse các sản phẩm
- Thử search, filter

### 2️ Thêm sản phẩm vào giỏ

- Click "Thêm vào giỏ"
- Vào trang giỏ hàng
- Thử cập nhật số lượng

### 3️ Login vào Backoffice

- Vào http://localhost:8000/backoffice
- Login với email bất kỳ
- Xem dashboard

### 4️ Quản lý sản phẩm (Admin)

- Vào trang Products
- Thử tạo sản phẩm mới
- Thử sửa/xóa sản phẩm

### 5️ Test API trực tiếp

- Vào http://localhost:3000/api/products
- Xem JSON response
- Thử các endpoints khác


##  Quick Commands Reference

```bash
# Setup
npm install

# Test
npm run test:api

# Run
npm start

# Backend only
npm run backend

# Frontend only
npm run frontend

# Pull latest
git pull origin main

# Commit
git add .
git commit -m "message"
git push origin main
```


