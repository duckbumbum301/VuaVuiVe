# 🚀 Hướng Dẫn Setup & Test Trang Gợi Ý ML-Powered

**Ngày:** 13 January 2026  
**Status:** ✅ Đã triển khai xong

---

## ✅ Các File Đã Được Tạo/Cập Nhật

### **1. HTML**

- ✅ `html/recommended.html` - Đã cập nhật với structure mới

### **2. CSS**

- ✅ `css/recommended-ml.css` - Styles cho ML-powered components

### **3. JavaScript**

- ✅ `js/recommended-ml.js` - Logic ML integration

---

## 🔧 Cấu Trúc Mới

### **Components Đã Triển Khai:**

#### 1. **Page Hero**

```html
- Tiêu đề: "Gợi ý dành riêng cho bạn" - Mô tả ngắn gọn - User interest chips
(dynamic từ ML)
```

#### 2. **Section Personal Recommendations**

```html
- Header với nút "Làm mới" (có icon rotate) - Products grid (4 columns) -
Loading skeletons
```

#### 3. **Section Similar Products**

```html
- Tự động load dựa trên top recommendation - Grid layout tương tự
```

#### 4. **Section Trending**

```html
- Sản phẩm phổ biến (weighted popularity)
```

---

## 🎨 Các Tính Năng UI/UX

### ✅ **Đã Implement:**

1. **Product Cards**

   - Badges động: 🔥 Bán chạy, 💛 Yêu thích, ✨ Gợi ý
   - ML reason text: "Bạn thường mua sản phẩm này"
   - Hover effects mượt mà
   - Responsive grid

2. **Loading States**

   - Skeleton screens khi load
   - Shimmer animation
   - Status messages

3. **Empty States**

   - Icon + message thân thiện
   - CTA button về trang chủ

4. **Toast Notifications**

   - Hiện khi add to cart
   - Auto dismiss sau 2.5s
   - Slide in animation

5. **Animations**
   - Fade in up cho cards (stagger effect)
   - Rotate icon khi click refresh
   - Smooth transitions

---

## 🔌 ML API Integration

### **Endpoints Đã Tích Hợp:**

```javascript
1. POST /api/recommend
   - Fetch personalized recommendations
   - Cache 30 phút
   - Fallback nếu fail

2. POST /api/similar
   - Fetch similar products
   - Dựa trên top recommendation

3. Batch processing
   - Multiple weights cho different sections
```

### **Fallback Logic:**

```
ML API Available?
  ├─ YES → Show ML recommendations
  └─ NO →
      ├─ Try cache (30 min)
      └─ Show trending products (rule-based)
```

---

## 🧪 Cách Test

### **Bước 1: Khởi động ML API**

```bash
# Terminal 1: Start ML API
cd VuaVuiVe_Recommender
python src/api.py

# Should see:
# 🎯 Recommendation API Server
# Running on http://localhost:5001
```

### **Bước 2: Khởi động Web Server**

```bash
# Terminal 2: Start JSON Server (backend)
cd e:\Nam3\TailLieuHocKi7\WEB2\VuaVuiVeNC
json-server --watch db.json --port 3000

# Hoặc nếu dùng Live Server trong VS Code:
# Right-click html/recommended.html → Open with Live Server
```

### **Bước 3: Truy cập trang**

```
http://localhost:5500/html/recommended.html
hoặc
http://127.0.0.1:5500/html/recommended.html
```

---

## 🔍 Test Cases

### **Test Case 1: Guest User (Chưa đăng nhập)**

**Expected:**

- Hiển thị chip: "🌱 Người dùng mới"
- Personal: 8 trending products
- Similar: Empty
- Trending: 4 trending products
- Status: "Đăng nhập để nhận gợi ý cá nhân hóa"

### **Test Case 2: Logged In User (Có lịch sử)**

**Expected:**

- Hiển thị user interest chips: 🥬 Rau củ, 🍎 Trái cây, etc.
- Personal: 8 ML recommendations với badges
- Similar: 6 products tương tự
- Trending: 8 popular products
- Status: "X sản phẩm được chọn lọc..."

### **Test Case 3: ML API Down**

**Expected:**

- Hiển thị fallback (trending products)
- Status: "Hiển thị gợi ý mặc định (ML API không khả dụng)"
- Không có crash
- UI vẫn đẹp

### **Test Case 4: Click "Làm mới"**

**Expected:**

- Clear cache
- Reload recommendations từ API
- Loading skeletons hiện lại
- Smooth transition

### **Test Case 5: Add to Cart**

**Expected:**

- Toast notification: "✅ Đã thêm vào giỏ hàng"
- Event được track vào localStorage
- Cart counter tăng (nếu có)

---

## 🎨 Visual Testing Checklist

### **Desktop (> 1024px)**

- ✅ 4 columns grid
- ✅ All badges visible
- ✅ Hover effects work
- ✅ Refresh button aligned right

### **Tablet (768px - 1024px)**

- ✅ 3 columns grid
- ✅ Cards maintain aspect ratio
- ✅ Button sizes appropriate

### **Mobile (< 768px)**

- ✅ 2 columns grid (> 480px)
- ✅ 1 column grid (< 480px)
- ✅ Refresh button full width
- ✅ Toast responsive
- ✅ Touch targets large enough

---

## 🐛 Common Issues & Solutions

### **Issue 1: ML API not responding**

**Symptoms:**

- Console error: "ML API failed"
- Fallback recommendations shown

**Solutions:**

1. Check ML API is running: `curl http://localhost:5001/health`
2. Check CORS settings in `src/api.py`
3. Check firewall/antivirus blocking port 5001

### **Issue 2: No products showing**

**Symptoms:**

- Empty state for all sections
- Console error: "Cannot read property 'map'"

**Solutions:**

1. Check JSON server is running: `http://localhost:3000/products`
2. Check `db.json` has products data
3. Check `apiListProducts()` function in `api.js`

### **Issue 3: Badges not showing**

**Symptoms:**

- All products without badges
- mlScore = 0

**Solutions:**

1. Check ML API returning scores
2. Check `enrichRecommendations()` mapping logic
3. Verify `getBadgeType()` thresholds

### **Issue 4: CSS not loading**

**Symptoms:**

- Ugly unstyled page
- Elements overlapping

**Solutions:**

1. Check `recommended-ml.css` path in HTML
2. Clear browser cache (Ctrl+Shift+R)
3. Check CSS file exists in `/css/` folder

---

## 📊 Performance Metrics

### **Expected Load Times:**

```
ML API Call: ~100-200ms
Product enrichment: ~50ms
DOM rendering: ~100ms
Total: ~250-350ms
```

### **Cache Hit Rate:**

- Target: > 70% (sau 30 phút đầu)

### **Memory Usage:**

- Cache storage: < 5MB
- Event tracking: < 1MB

---

## 🎯 Next Steps (Optional Enhancements)

### **Phase 2 Features:**

1. **Product Mapping Table**

   ```javascript
   // Create proper ID mapping
   const PRODUCT_MAPPING = {
     vvv_001: 24852, // VVV ID → Instacart ID
     // ... full mapping
   };
   ```

2. **Advanced Analytics**

   ```javascript
   // Track more events
   -impression - click - add_to_cart - purchase - time_on_page;
   ```

3. **A/B Testing**

   ```javascript
   // Split traffic 50/50
   if (Math.random() < 0.5) {
     // Show ML recommendations
   } else {
     // Show rule-based
   }
   ```

4. **Personalized Badges**
   ```javascript
   // Custom badges per user
   -"Bạn đã mua 5 lần" - "Giảm 20% cho bạn" - "Sắp hết hàng";
   ```

---

## 🔐 Security Notes

### **API Security:**

- ML API chỉ chạy localhost (development)
- Production: Cần setup authentication
- Rate limiting để tránh abuse

### **Data Privacy:**

- Không hiển thị user_id ra frontend
- ML scores chỉ dùng internal
- Event tracking anonymous

---

## 📝 Code Quality Checklist

- ✅ ES6 modules
- ✅ Async/await (không callback hell)
- ✅ Error handling với try/catch
- ✅ Loading states
- ✅ Responsive design
- ✅ Accessibility (aria-live, alt text)
- ✅ Comments đầy đủ
- ✅ No console.log spam

---

## 🎓 Learning Resources

### **ML Concepts:**

- NMF (Non-negative Matrix Factorization)
- Collaborative Filtering
- Co-occurrence Analysis
- Hybrid Recommender Systems

### **Frontend Patterns:**

- Cache-first strategy
- Optimistic UI updates
- Skeleton screens
- Stagger animations

---

## ✅ Deployment Checklist

### **Pre-deploy:**

- [ ] Test trên Chrome, Firefox, Safari
- [ ] Test responsive trên mobile thật
- [ ] Performance audit (Lighthouse)
- [ ] Accessibility audit (Wave/axe)

### **Deploy:**

- [ ] Build production assets
- [ ] Minify CSS/JS
- [ ] Setup CDN cho images
- [ ] Configure API endpoints

### **Post-deploy:**

- [ ] Monitor error rates
- [ ] Check analytics working
- [ ] Verify ML API stable
- [ ] Collect user feedback

---

## 🆘 Support

### **Khi gặp vấn đề:**

1. **Check Console**

   - Mở DevTools (F12)
   - Tab Console
   - Copy error messages

2. **Check Network**

   - Tab Network
   - Filter: XHR
   - Check API responses

3. **Debug Steps**
   ```javascript
   // Add debug logs
   console.log("🔍 Debug:", {
     user,
     mlData,
     products,
   });
   ```

---

**Chúc bạn thành công! 🎉**

Nếu có câu hỏi, check lại [FRONTEND_DESIGN_GUIDE.md](FRONTEND_DESIGN_GUIDE.md) để xem chi tiết implementation.
