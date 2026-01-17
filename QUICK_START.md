# 🚀 Quick Start - Vựa Vui Vẻ (Full Stack + ML)

**Cập nhật:** 13 January 2026  
**Version:** 2.0 (với ML Recommender)

---

## ⚡ Khởi Động Nhanh

### **Chỉ cần 1 lệnh:**

```bash
# Windows:
start-all.bat

# Hoặc double-click file start-all.bat
```

**Tất cả services sẽ tự động chạy:**

- ✅ VNPay Backend (Port 8888)
- ✅ API Backend (Port 3000)
- ✅ Frontend (Port 8000)
- ✅ ML Recommender API (Port 5001) 🆕

---

## 🌐 Truy Cập

### **Frontend Pages:**

- 🏠 **Trang chủ:** http://localhost:8000/Home/Hôm/index.html
- 🛒 **Sản phẩm:** http://localhost:8000/html/index.html
- 🛍️ **Giỏ hàng:** http://localhost:8000/html/cart.html
- 🎁 **Gợi ý ML:** http://localhost:8000/html/recommended.html 🆕
- 👤 **Tài khoản:** http://localhost:8000/html/account.html
- 🏢 **Backoffice:** http://localhost:8000/backoffice/

### **API Endpoints:**

- 📦 **Products:** http://localhost:3000/products
- 👥 **Users:** http://localhost:3000/users
- 📝 **Orders:** http://localhost:3000/orders
- 💳 **VNPay:** http://localhost:8888/order/create_payment_url
- 🤖 **ML Recommender:** http://localhost:5001/health 🆕

---

## 📋 Yêu Cầu Hệ Thống

### **Bắt buộc:**

- ✅ Node.js (v16+)
- ✅ npm hoặc yarn

### **Tùy chọn (cho ML):**

- ⭐ Python (v3.8+)
- ⭐ pip

**Không có Python?** → ML sẽ tự động bỏ qua, website vẫn chạy bình thường!

---

## 🔧 Setup Lần Đầu

### **Bước 1: Clone/Download Project**

```bash
cd e:\Nam3\TailLieuHocKi7\WEB2\VuaVuiVeNC
```

### **Bước 2: Install Dependencies**

```bash
# Node.js dependencies (tự động trong start-all.bat)
npm install
cd vnpay_nodejs
npm install
cd ..
```

### **Bước 3 (Optional): Setup ML**

```bash
# Cài Python dependencies
cd VuaVuiVe_Recommender
pip install -r requirements.txt

# Kiểm tra model đã có chưa
dir models\nmf_model.pkl
```

**Nếu chưa có model:** Train bằng notebooks:

1. Mở `VuaVuiVe_Recommender/notebooks/01_data_cleaning.ipynb`
2. Chạy `02_feature_engineering.ipynb`
3. Chạy `03_train_model.ipynb`

### **Bước 4: Chạy!**

```bash
start-all.bat
```

---

## 🎯 Các Services Chạy

Sau khi chạy `start-all.bat`, bạn sẽ thấy **4 cửa sổ terminal** mở ra:

### **Terminal 1: VNPay Backend (Màu vàng)**

```
Port: 8888
Status: Server running at http://localhost:8888
```

### **Terminal 2: API Backend (Màu xanh dương)**

```
Port: 3000
Status: JSON Server running
Resources: /products, /users, /orders
```

### **Terminal 3: Frontend (Màu hồng)**

```
Port: 8000
Status: Serving static files
Root: e:\Nam3\TailLieuHocKi7\WEB2\VuaVuiVeNC
```

### **Terminal 4: ML API (Màu đỏ) 🆕**

```
Port: 5001
Status: 🎯 Recommendation API Server
Endpoints: /health, /api/recommend, /api/similar
```

---

## ✅ Kiểm Tra Services

### **Cách 1: Trong Terminal**

```bash
# Tất cả ports đang chạy
netstat -ano | findstr ":8888 :3000 :8000 :5001"
```

### **Cách 2: Browser**

```bash
# Health checks
curl http://localhost:3000/products
curl http://localhost:5001/health
```

### **Cách 3: Tự động**

File `start-all.bat` sẽ tự động mở browser với các trang:

- Trang chủ
- Gợi ý ML
- Backoffice

---

## 🐛 Troubleshooting

### **❌ ML API không chạy**

**Triệu chứng:**

- Cảnh báo: "Python chưa được cài đặt"
- Terminal 4 không mở
- ML API bỏ qua

**Giải pháp:**

1. Cài Python: https://www.python.org/
2. Cài dependencies: `pip install -r VuaVuiVe_Recommender/requirements.txt`
3. Chạy lại `start-all.bat`

**Tạm thời:**

- Website vẫn chạy bình thường
- Trang gợi ý sẽ hiển thị sản phẩm trending (fallback)

---

### **❌ Port đã bị sử dụng**

**Triệu chứng:**

- Error: "EADDRINUSE"
- Service không start được

**Giải pháp:**

```bash
# Dừng tất cả Node.js processes
taskkill /F /IM node.exe

# Dừng Python
taskkill /F /IM python.exe

# Chạy lại
start-all.bat
```

---

### **❌ Dependencies không đúng**

**Triệu chứng:**

- Module not found
- Import errors

**Giải pháp:**

```bash
# Node.js
npm install
cd vnpay_nodejs && npm install && cd ..

# Python
cd VuaVuiVe_Recommender
pip install -r requirements.txt --upgrade
cd ..
```

---

## 🛑 Dừng Services

### **Cách 1: Đóng terminals**

- Đóng 4 cửa sổ terminal đã mở

### **Cách 2: Ctrl+C**

- Nhấn `Ctrl+C` trong mỗi terminal

### **Cách 3: Task Manager**

- Mở Task Manager
- Tìm `node.exe` và `python.exe`
- End Task

---

## 📊 Monitoring

### **Check Health:**

```bash
# API Backend
curl http://localhost:3000/products

# ML API
curl http://localhost:5001/health

# Expected response:
{
  "status": "ok",
  "message": "Recommendation API is running"
}
```

### **Check Logs:**

Xem trong các terminal windows:

- VNPay: Màu vàng
- API: Màu xanh dương
- Frontend: Màu hồng
- ML API: Màu đỏ

---

## 🎓 Tips & Tricks

### **1. Development Workflow**

```bash
# Chạy tất cả
start-all.bat

# Edit code trong VS Code
# Browser auto-reload (nếu dùng Live Server)

# Xem logs trong terminals
```

### **2. Testing ML API**

```bash
# Test health
curl http://localhost:5001/health

# Test recommendations (PowerShell)
$body = @{
  user_id = 1
  n = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5001/api/recommend" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### **3. Quick Restart**

```bash
# Dừng
taskkill /F /IM node.exe
taskkill /F /IM python.exe

# Chạy lại
start-all.bat
```

---

## 📁 Project Structure

```
VuaVuiVeNC/
├── start-all.bat           # 🚀 Launch script
├── package.json            # Node dependencies
├── db.json                 # API database
│
├── html/                   # Frontend pages
│   ├── index.html
│   ├── cart.html
│   ├── recommended.html    # 🆕 ML-powered page
│   └── ...
│
├── css/                    # Stylesheets
│   ├── style.css
│   └── recommended-ml.css  # 🆕 ML page styles
│
├── js/                     # JavaScript
│   ├── api.js
│   ├── cart.js
│   └── recommended-ml.js   # 🆕 ML integration
│
├── backoffice/             # Admin panel
│   └── index.html
│
├── vnpay_nodejs/           # VNPay integration
│   └── app.js
│
└── VuaVuiVe_Recommender/   # 🆕 ML System
    ├── src/
    │   ├── api.py          # Flask API
    │   └── recommender.py  # ML logic
    ├── models/             # Trained models
    ├── data/               # Datasets
    └── notebooks/          # Jupyter notebooks
```

---

## 🎯 Features

### **Customer-facing:**

- ✅ Browse products
- ✅ Add to cart
- ✅ Checkout with VNPay
- ✅ View account/orders
- 🆕 **ML-powered recommendations**
  - Personal recommendations
  - Similar products
  - Trending items

### **Admin:**

- ✅ Dashboard analytics
- ✅ Product management
- ✅ Order management
- ✅ User management

### **ML System:** 🆕

- ✅ Hybrid recommender (NMF + Co-occurrence + Popularity)
- ✅ Real-time API
- ✅ Fallback to trending
- ✅ Cache optimization

---

## 🚀 Next Steps

1. **Test the app:**

   - Browse products
   - Add to cart
   - Check ML recommendations
   - Test checkout

2. **Customize:**

   - Update product data in `db.json`
   - Customize styles in CSS
   - Tune ML weights in `recommended-ml.js`

3. **Deploy:**
   - Setup production server
   - Configure domain
   - Setup SSL
   - Deploy ML API

---

## 📞 Support

### **Documentation:**

- [FRONTEND_DESIGN_GUIDE.md](VuaVuiVe_Recommender/FRONTEND_DESIGN_GUIDE.md)
- [SETUP_GUIDE.md](VuaVuiVe_Recommender/SETUP_GUIDE.md)
- [IMPLEMENTATION_SUMMARY.md](VuaVuiVe_Recommender/IMPLEMENTATION_SUMMARY.md)

### **Common Issues:**

- Port conflicts → Kill processes
- ML not working → Install Python
- Dependencies errors → Reinstall
- Browser cache → Hard refresh (Ctrl+Shift+R)

---

**Happy Coding! 🎉**

Chạy `start-all.bat` và tận hưởng! 🚀
