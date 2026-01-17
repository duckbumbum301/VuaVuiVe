# 🚀 Hướng dẫn sử dụng Model ML với VVV

## ✅ Đã hoàn thành

Hệ thống mapping VVV ↔ Instacart đã sẵn sàng! Model ML Instacart (đã train) có thể dùng NGAY cho VVV.

---

## 📂 Cấu trúc Files

```
VuaVuiVe_Recommender/
├── mappings/
│   └── vvv_instacart_mapping.json    # Mapping categories VVV ↔ Instacart
├── src/
│   ├── api.py                        # ✅ ĐÃ CẬP NHẬT - API với VVV adapter
│   ├── vvv_adapter.py                # ✅ MỚI - Adapter VVV ↔ Instacart
│   └── recommender.py                # Giữ nguyên - Instacart model
├── test_vvv_integration.py           # ✅ MỚI - Test script
└── models/
    └── nmf_model.pkl                 # Model đã train sẵn
```

---

## 🔧 Cách hoạt động

### **Quy trình ánh xạ:**

```
VVV User (ID: 4)
    ↓
Lịch sử mua: ["100", "110", "112"]  (Rau muống, Cà rốt, Bí đỏ)
    ↓
Mapping → Instacart: [13176, 24852, 27344, 47766, ...]  (Rau củ category)
    ↓
Instacart Model: Gợi ý dựa trên 200K+ users
    ↓
Recommendations: [(13176, 42.5), (38928, 35.2), ...]
    ↓
Mapping ngược → VVV: Product ID 100-199 (category veg/leaf, veg/root)
    ↓
Kết quả: VVV products với scores và reasons
```

### **Điểm mạnh:**

✅ **Không cần train lại** - Dùng ngay model đã có
✅ **Học từ 200K users** - Patterns chung về grocery shopping
✅ **Category-based** - Flexible, dễ update
✅ **Auto fallback** - Popular products khi cold start

---

## 🧪 Test trước khi chạy

### **Bước 1: Test adapter**

```bash
cd VuaVuiVe_Recommender
python test_vvv_integration.py
```

**Kết quả mong đợi:**

```
==============================================================
  VVV-INSTACART INTEGRATION TEST
==============================================================

TEST 1: VVV ADAPTER
✓ Loaded 97 products, 1516 orders, 110 users
✓ Loaded mapping for 41 categories

[Test 1.1] Get user purchase history
  User 4 (Nguyễn Văn A): 2 products purchased
  User 5 (Trần Thị B): 3 products purchased

[Test 1.2] VVV → Instacart mapping
  Mapped to Instacart: [13176, 24852, 27344, ...]

[Test 1.3] Instacart → VVV mapping
  Top 5 VVV recommendations:
    1. Rau muống (500g) (Score: 29.75) - Khách hàng thường mua cùng
    2. Cải bẹ xanh (500g) (Score: 26.74) - Khách hàng thường mua cùng
    ...

✓ ALL TESTS PASSED!
```

---

## 🚀 Khởi động ML API

### **Cách 1: Tự động (khuyên dùng)**

```bash
# Từ root project
./start-all.bat
```

ML API sẽ tự động chạy trên **port 5001** nếu có Python.

### **Cách 2: Thủ công**

```bash
cd VuaVuiVe_Recommender
python src/api.py
```

**Output:**

```
🔥 Loading recommendation model...
✓ Model loaded successfully!
🔄 Loading VVV-Instacart adapter...
✓ Loaded 97 products, 1516 orders, 110 users
✓ Loaded mapping for 41 categories
✓ Adapter loaded successfully!

 * Running on http://127.0.0.1:5001
```

---

## 🧪 Test API

### **Test 1: Health check**

```bash
curl http://localhost:5001/health
```

**Response:**

```json
{
  "status": "ok",
  "message": "Recommendation API is running"
}
```

### **Test 2: Recommendations cho VVV user**

```bash
curl -X POST http://localhost:5001/api/recommend \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": 4, \"n\": 10}"
```

**Response:**

```json
{
  "user_id": 4,
  "recommendations": [
    {
      "product_id": 100,
      "score": 29.75,
      "name": "Rau muống (500g)",
      "price": 15000,
      "image": "../images/VEG/leaf/raumuong.jpg",
      "category": "veg/leaf",
      "reason": "Khách hàng thường mua cùng"
    },
    {
      "product_id": 110,
      "score": 26.74,
      "name": "Cà rốt (500g)",
      "price": 15000,
      "image": "../images/VEG/root/Cà Rốt.jpg",
      "category": "veg/root",
      "reason": "Khách hàng thường mua cùng"
    }
  ],
  "count": 10,
  "method": "hybrid_instacart_mapping"
}
```

### **Test 3: Cold start user (no history)**

```bash
curl -X POST http://localhost:5001/api/recommend \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": 999, \"n\": 5}"
```

**Response:** Popular products (fallback)

---

## 🌐 Test từ Frontend

1. Mở: http://localhost:8000/html/recommended.html
2. Đăng nhập với user có lịch sử (user ID 4-8)
3. Xem 3 sections:
   - "Tiếp nối những gì bạn thích" - Personal recs
   - "Sản phẩm tương tự" - Similar items
   - "Mua nhiều tại Vựa Vui Vẻ" - Trending

**Kỳ vọng:**

- ✅ Hiển thị VVV products (không phải Instacart)
- ✅ Prices và images chính xác
- ✅ Reasons human-friendly ("Bạn thường mua", "Khách hàng thường mua cùng")
- ✅ Badges dựa trên ML scores

---

## 🔍 Debug

### **Problem 1: API lỗi khi start**

**Lỗi:** `FileNotFoundError: vvv_instacart_mapping.json`

**Fix:**

```bash
# Kiểm tra file mapping tồn tại
ls VuaVuiVe_Recommender/mappings/vvv_instacart_mapping.json
```

### **Problem 2: Recommendations trống**

**Debug:**

```python
# Check response
response['debug']
# Output:
{
  "vvv_history_count": 0,     # ← User không có lịch sử → Cold start
  "instacart_proxy_count": 0,
  "vvv_candidates_count": 0
}
```

**Fix:** User cold start → Trả về popular products (đã implement)

### **Problem 3: Frontend không hiển thị**

**Check:**

1. ML API đang chạy: http://localhost:5001/health
2. Console log trong browser (F12)
3. Network tab - Check API calls

**Common issues:**

- CORS error → API đã enable CORS, restart API
- 404 error → Check ML_API_URL trong recommended-ml.js
- Timeout → Tăng timeout trong fetchMLRecommendations()

---

## 📊 Monitoring

### **Check API logs**

API sẽ in ra terminal:

```
🔥 Loading recommendation model...
✓ Loaded 97 products, 1516 orders, 110 users
✓ Adapter loaded successfully!

[Request] POST /api/recommend
  User: 4
  History: 2 products
  Instacart proxy: 10 products
  Recommendations: 15 candidates → 10 final
```

### **Performance metrics**

- **Cold start:** ~100ms (popular products)
- **With history:** ~200-300ms (mapping + model inference)
- **Cache:** 30 minutes (frontend)

---

## 🔄 Cập nhật khi có đơn hàng mới

### **Option 1: Không làm gì (Recommended)**

Adapter tự động load orders.json mỗi lần restart API.

**Khi nào restart:**

- Sau 1 tuần có nhiều đơn mới
- Hoặc restart hàng ngày qua cron job

### **Option 2: Hot reload**

Thêm endpoint `/api/reload`:

```python
@app.route('/api/reload', methods=['POST'])
def reload_data():
    global adapter
    adapter = VVVInstacartAdapter(VVV_DATA_DIR, MAPPING_FILE)
    return jsonify({'status': 'reloaded'})
```

Call từ backoffice sau khi có đơn mới:

```javascript
fetch("http://localhost:5001/api/reload", { method: "POST" });
```

---

## 🎯 Nâng cấp sau này

### **Phase 1: Đang dùng (hiện tại)**

✅ Category-based mapping
✅ Instacart model cho patterns chung
✅ VVV popularity fallback

### **Phase 2: Khi có 500+ orders VVV**

- Train incremental model với VVV data
- Combine: 70% VVV model + 30% Instacart patterns

### **Phase 3: Khi có 2000+ orders**

- Train dedicated VVV model
- Instacart chỉ dùng cho cold start

---

## ✅ Checklist Triển khai

- [x] Tạo mapping file
- [x] Implement VVV adapter
- [x] Update API với adapter
- [x] Update frontend (recommended-ml.js)
- [x] Test adapter
- [x] Test API endpoints
- [ ] **→ Test từ browser**
- [ ] **→ Monitor trong 1 tuần**
- [ ] **→ Thu thập feedback users**

---

## 🆘 Support

Nếu có vấn đề:

1. Check logs trong terminal ML API
2. Run test script: `python test_vvv_integration.py`
3. Check debug info trong API response
4. Kiểm tra orders.json có data không

**Contact:** Backend developer / ML engineer
