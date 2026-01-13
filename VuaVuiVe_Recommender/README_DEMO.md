# 🎯 Demo Hệ Thống Gợi Ý Sản Phẩm - ML Recommender

## 📋 Tổng quan

Trang web demo đơn giản để trải nghiệm hệ thống gợi ý sản phẩm dựa trên Machine Learning với **Hybrid Recommender System** kết hợp 3 phương pháp:

- 🤖 **Collaborative Filtering (NMF)**: Gợi ý dựa trên hành vi người dùng tương tự
- 🛒 **Co-occurrence (Basket Analysis)**: Gợi ý sản phẩm thường mua cùng nhau
- ⭐ **Popularity-based**: Gợi ý sản phẩm phổ biến

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Khởi động API Server

Mở terminal và chạy lệnh:

```bash
# Kích hoạt virtual environment (nếu có)
.venv\Scripts\activate

# Chạy API server
cd VuaVuiVe_Recommender
python src/api.py
```

Server sẽ chạy tại: `http://localhost:5001`

### Bước 2: Mở trang web demo

Mở file `demo.html` bằng trình duyệt web (Chrome, Firefox, Edge, etc.)

Hoặc dùng Live Server trong VS Code:

- Click phải vào `demo.html` → "Open with Live Server"

### Bước 3: Trải nghiệm các tính năng

#### 📌 Tính năng 1: Gợi Ý Cho Người Dùng

1. Nhập **User ID** (từ 1 đến 206209)
2. Chọn **số lượng gợi ý** (5, 10, 15, 20 sản phẩm)
3. Tùy chọn **lọc bỏ sản phẩm đã mua**
4. Click **"🚀 Lấy Gợi Ý"**

Ví dụ User IDs để test:

- User ID: 1
- User ID: 100
- User ID: 1000
- User ID: 50000

#### 📌 Tính năng 2: Gợi Ý Từ Giỏ Hàng

1. Nhập **User ID**
2. Thêm các **Product ID** vào giỏ hàng
3. Click **"🎁 Gợi Ý Từ Giỏ Hàng"**

Ví dụ Product IDs để test:

- 24852 (Banana)
- 13176 (Bag of Organic Bananas)
- 27344
- 21903

## 🎨 Tính Năng Nổi Bật

### UI/UX

- ✨ Giao diện hiện đại với gradient màu sắc
- 🎭 Animations mượt mà
- 📱 Responsive design (hỗ trợ mobile)
- 🎯 Real-time feedback

### Chức năng

- 👤 Gợi ý cá nhân hóa cho từng user
- 🛒 Gợi ý dựa trên giỏ hàng hiện tại
- 📊 Hiển thị điểm số (score) cho mỗi gợi ý
- 🏆 Xếp hạng sản phẩm từ cao đến thấp
- ⚡ Loading states và error handling

### Thống kê

- 📈 User ID đang xem
- 🔢 Số lượng sản phẩm gợi ý
- 🏷️ Loại gợi ý (User/Cart)

## 🛠️ Kiến Trúc Kỹ Thuật

```
┌─────────────┐         HTTP POST          ┌──────────────┐
│             │  ───────────────────────>  │              │
│  demo.html  │    /api/recommend          │   API Server │
│  (Frontend) │                            │  (Flask)     │
│             │  <───────────────────────  │              │
└─────────────┘         JSON Response      └──────────────┘
                                                   │
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │   ML Model   │
                                           │  (NMF + CF)  │
                                           └──────────────┘
```

## 📡 API Endpoints

### 1. Health Check

```bash
GET http://localhost:5001/health
```

### 2. Get Recommendations

```bash
POST http://localhost:5001/api/recommend

Body:
{
    "user_id": 1,
    "cart_items": [24852, 13176],  // optional
    "n": 10,
    "filter_purchased": true
}

Response:
{
    "user_id": 1,
    "recommendations": [
        {"product_id": 27344, "score": 40.28},
        {"product_id": 21903, "score": 35.67},
        ...
    ],
    "count": 10
}
```

## 🐛 Xử Lý Lỗi

### Lỗi: "Failed to fetch"

**Nguyên nhân**: API server chưa chạy hoặc CORS chưa được cấu hình

**Giải pháp**:

1. Đảm bảo server đang chạy: `python src/api.py`
2. Kiểm tra API_URL trong demo.html: `http://localhost:5001`
3. Kiểm tra CORS đã được enable trong `src/api.py`

### Lỗi: "User not found" hoặc score = 0

**Nguyên nhân**: User ID không có trong training data

**Giải pháp**:

- Thử User ID khác (từ 1-206209)
- Hệ thống sẽ fallback về popularity-based recommendations

### Lỗi: "Model không tồn tại"

**Nguyên nhân**: Chưa train model

**Giải pháp**:
Chạy notebook `03_train_model.ipynb` để train model trước

## 📊 Dataset

- **Users**: 206,209 người dùng
- **Products**: 49,688 sản phẩm
- **Orders**: 3,421,083 đơn hàng
- **Data source**: Instacart Market Basket Analysis

## 🎓 Thuật Toán ML

### Hybrid Score Calculation

```python
score = w_cf * cf_score + w_basket * basket_score + w_pop * pop_score
```

Mặc định:

- `w_cf = 0.5` (50% từ CF)
- `w_basket = 0.3` (30% từ basket analysis)
- `w_pop = 0.2` (20% từ popularity)

## 🔧 Tùy Chỉnh

### Thay đổi API URL

Trong `demo.html`, dòng 352:

```javascript
const API_URL = "http://localhost:5001";
```

### Thay đổi weights cho hybrid model

Trong `src/recommender.py`, function `recommend()`:

```python
w_cf=0.5,      # CF weight
w_basket=0.3,  # Basket weight
w_pop=0.2      # Popularity weight
```

## 🌟 Demo Examples

### Example 1: User Recommendations

```
User ID: 1
Number of recommendations: 10
Filter purchased: Yes
→ Kết quả: 10 sản phẩm phù hợp với sở thích của user
```

### Example 2: Cart-based Recommendations

```
User ID: 100
Cart items: [24852, 13176, 21903]
→ Kết quả: Gợi ý các sản phẩm thường mua cùng với bananas
```

## 📝 Notes

- Model được train trên Instacart dataset (~3M orders)
- Sử dụng Matrix Factorization (NMF) với 20 latent factors
- Co-occurrence matrix với min support = 5
- Tất cả scores được normalize về range [0, 100]

## 🚀 Next Steps

Để tích hợp vào production:

1. **Deploy API**: Deploy lên cloud (Heroku, AWS, GCP)
2. **Add Product Info**: Kết nối với database để hiển thị tên, hình ảnh sản phẩm
3. **A/B Testing**: Test các weights khác nhau
4. **Monitoring**: Add logging và metrics
5. **Caching**: Cache results cho performance
6. **Authentication**: Add user authentication

## 📞 Support

Nếu có vấn đề, kiểm tra:

1. ✅ API server đang chạy
2. ✅ Model đã được train
3. ✅ CORS enabled
4. ✅ Port 5001 chưa bị chiếm

---

**Created by**: ML Expert with 30+ years experience 🎓
**Tech Stack**: Python, Flask, Scikit-learn, NMF, HTML/CSS/JavaScript
**Version**: 1.0.0
