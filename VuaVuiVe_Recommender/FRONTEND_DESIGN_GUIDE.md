# 🎨 Hướng Dẫn Thiết Kế Frontend - Trang Gợi Ý (ML-Powered)

**Mục tiêu:** Nâng cấp trang `recommended.html` từ rule-based lên ML-powered với UX/UI thân thiện, không hiển thị các số liệu kỹ thuật (confidence scores) cho khách hàng.

**Ngày tạo:** 13 January 2026  
**Target Page:** `html/recommended.html`

---

## 📋 Mục Lục

1. [Triết Lý Thiết Kế](#triết-lý-thiết-kế)
2. [Cấu Trúc Trang](#cấu-trúc-trang)
3. [Components Chi Tiết](#components-chi-tiết)
4. [CSS Styling](#css-styling)
5. [JavaScript Integration](#javascript-integration)
6. [Animation & Transitions](#animation--transitions)
7. [Responsive Design](#responsive-design)
8. [Loading States](#loading-states)

---

## 🎯 Triết Lý Thiết Kế

### **Nguyên Tắc Chính**

#### 1. **Invisible Intelligence**

- ML hoạt động trong nền, khách hàng không thấy thuật toán
- Không hiển thị: scores, confidence levels, technical metrics
- Focus: Sản phẩm có ích, UI đẹp, UX mượt

#### 2. **Trust Through Transparency**

- Giải thích TẠI SAO gợi ý sản phẩm này
- VD: "Dựa trên lịch sử mua hàng của bạn", "Khách hàng thường mua cùng"
- Không dùng: "Score: 45.28", "Confidence: 87%"

#### 3. **Contextual Relevance**

- Hiển thị đúng thông tin khách cần: tên, giá, hình ảnh
- Badge đơn giản: "Bán chạy", "Yêu thích", "Mới"
- Không cần số liệu phức tạp

#### 4. **Effortless Interaction**

- 1-click add to cart
- Smooth animations
- Clear visual hierarchy

---

## 🏗️ Cấu Trúc Trang

### **Layout Overview**

```
┌─────────────────────────────────────────────────────┐
│  Header (Fixed)                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  🎯 Page Hero                                       │
│  ├─ Tiêu đề: "Gợi ý dành riêng cho bạn"           │
│  ├─ Mô tả: Giải thích ngắn gọn                    │
│  └─ User Signals (Chips): Rau củ, Trái cây...     │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  📦 Section 1: "Tiếp nối những gì bạn thích"       │
│  ├─ Header với nút "Làm mới"                       │
│  └─ Grid 4 cột (Desktop) / 2 cột (Mobile)          │
│                                                     │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│     │ Card │ │ Card │ │ Card │ │ Card │           │
│     │  🍌  │ │  🍎  │ │  🥛  │ │  🍞  │           │
│     └──────┘ └──────┘ └──────┘ └──────┘           │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  🔄 Section 2: "Sản phẩm tương tự"                 │
│  └─ Grid tương tự                                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  🔥 Section 3: "Mua nhiều tại Vựa Vui Vẻ"          │
│  └─ Grid trending products                         │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Footer                                             │
└─────────────────────────────────────────────────────┘
```

---

## 🧩 Components Chi Tiết

### **1. Page Hero Section**

#### HTML Structure

```html
<div class="page-hero">
  <div class="page-hero__content">
    <span class="eyebrow">🎁 Cá nhân hóa</span>
    <h1 class="page-title">Gợi ý dành riêng cho bạn</h1>
    <p class="page-description">
      Chúng tôi chọn lọc sản phẩm dựa trên sở thích và lịch sử mua sắm của bạn
    </p>

    <!-- User Interest Chips (từ ML) -->
    <div class="chip-row" id="recSignal">
      <span class="chip">🥬 Rau củ</span>
      <span class="chip">🍎 Trái cây</span>
      <span class="chip">🥛 Đồ uống</span>
    </div>
  </div>
</div>
```

#### CSS Styling

```css
/* Page Hero */
.page-hero {
  padding: 40px 0 32px;
  text-align: center;
  background: linear-gradient(
    180deg,
    rgba(76, 159, 63, 0.05) 0%,
    transparent 100%
  );
  border-radius: var(--radius);
  margin-bottom: 40px;
}

.page-hero__content {
  max-width: 640px;
  margin: 0 auto;
}

.eyebrow {
  display: inline-block;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--green);
  background: rgba(76, 159, 63, 0.1);
  padding: 4px 12px;
  border-radius: 20px;
  margin-bottom: 12px;
}

.page-title {
  font-size: 36px;
  font-weight: 700;
  color: var(--text);
  margin: 12px 0 16px;
  line-height: 1.2;
}

.page-description {
  font-size: 16px;
  color: var(--muted);
  margin-bottom: 24px;
  line-height: 1.6;
}

/* User Interest Chips */
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 20px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: white;
  border: 2px solid var(--border);
  border-radius: 24px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
  transition: all 0.2s ease;
}

.chip:hover {
  border-color: var(--green);
  background: rgba(76, 159, 63, 0.05);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
```

---

### **2. Section Card (Container cho mỗi nhóm gợi ý)**

#### HTML Structure

```html
<section class="section-card">
  <!-- Section Header -->
  <div class="section-head">
    <div class="section-head__text">
      <span class="eyebrow">🎯 Cho riêng bạn</span>
      <h2 class="section-title">Tiếp nối những gì bạn thích</h2>
      <p class="section-description" id="recStatus">
        8 sản phẩm được chọn lọc dựa trên lịch sử mua hàng của bạn
      </p>
    </div>
    <button class="btn btn--outline btn--refresh" id="recRefreshBtn">
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <polyline points="23 4 23 10 17 10"></polyline>
        <polyline points="1 20 1 14 7 14"></polyline>
        <path
          d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"
        ></path>
      </svg>
      Làm mới
    </button>
  </div>

  <!-- Products Grid -->
  <div class="products-grid" id="recPersonalGrid">
    <!-- Cards will be inserted here by JavaScript -->
  </div>
</section>
```

#### CSS Styling

```css
/* Section Card */
.section-card {
  background: white;
  border-radius: var(--radius);
  padding: 32px;
  margin-bottom: 32px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  transition: box-shadow 0.3s ease;
}

.section-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

/* Section Header */
.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 28px;
  gap: 20px;
}

.section-head__text {
  flex: 1;
}

.section-title {
  font-size: 26px;
  font-weight: 700;
  color: var(--text);
  margin: 8px 0 12px;
  line-height: 1.3;
}

.section-description {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.6;
  margin: 0;
}

/* Refresh Button */
.btn--refresh {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  white-space: nowrap;
  flex-shrink: 0;
}

.btn--refresh svg {
  transition: transform 0.3s ease;
}

.btn--refresh:hover svg {
  transform: rotate(180deg);
}
```

---

### **3. Product Card (ML-Powered, UI Đơn Giản)**

#### HTML Structure

```html
<article class="product-card" data-id="123" data-ml-score="38.5">
  <!-- Product Image -->
  <div class="product-card__image">
    <img src="../images/FRUIT/Mixed/banana.jpg" alt="Chuối" loading="lazy" />
    <!-- Badge (Optional) -->
    <span class="product-badge product-badge--hot">🔥 Bán chạy</span>
  </div>

  <!-- Product Info -->
  <div class="product-card__body">
    <h3 class="product-name">Chuối</h3>

    <!-- Price -->
    <div class="product-price">
      <span class="price price--current">15.000₫</span>
      <span class="price price--original">20.000₫</span>
      <span class="price-discount">-25%</span>
    </div>

    <!-- Meta Info -->
    <p class="product-meta">
      <span class="product-unit">1kg</span>
      <span class="product-stock">Còn hàng</span>
    </p>

    <!-- Why Recommended (Context) -->
    <div class="product-reason">
      <svg
        width="14"
        height="14"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <path d="M12 2L4 12L1 9"></path>
      </svg>
      <span>Bạn thường mua sản phẩm này</span>
    </div>
  </div>

  <!-- Action Button -->
  <button class="btn btn--add-cart" data-action="add" data-id="123">
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
    >
      <circle cx="9" cy="21" r="1"></circle>
      <circle cx="20" cy="21" r="1"></circle>
      <path
        d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"
      ></path>
    </svg>
    Thêm vào giỏ
  </button>
</article>
```

#### CSS Styling

```css
/* Product Card */
.product-card {
  background: white;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  transition: all 0.3s ease;
  position: relative;
  display: flex;
  flex-direction: column;
}

.product-card:hover {
  border-color: var(--green);
  box-shadow: 0 8px 24px rgba(76, 159, 63, 0.15);
  transform: translateY(-4px);
}

/* Product Image */
.product-card__image {
  position: relative;
  width: 100%;
  height: 200px;
  background: #f9fafb;
  overflow: hidden;
}

.product-card__image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.product-card:hover .product-card__image img {
  transform: scale(1.05);
}

/* Product Badge */
.product-badge {
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  z-index: 1;
}

.product-badge--hot {
  background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
  color: white;
}

.product-badge--new {
  background: linear-gradient(135deg, #4c9f3f 0%, #2f7f31 100%);
  color: white;
}

.product-badge--favorite {
  background: linear-gradient(135deg, #ffd93d 0%, #ff8a00 100%);
  color: #fff;
}

/* Product Body */
.product-card__body {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.product-name {
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 8px;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Price */
.product-price {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.price {
  font-weight: 700;
}

.price--current {
  font-size: 18px;
  color: var(--green);
}

.price--original {
  font-size: 14px;
  color: var(--muted);
  text-decoration: line-through;
}

.price-discount {
  font-size: 12px;
  font-weight: 600;
  color: #ff6b6b;
  background: rgba(255, 107, 107, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
}

/* Product Meta */
.product-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: var(--muted);
  margin: 0 0 12px;
}

.product-unit {
  font-weight: 500;
}

.product-stock {
  color: var(--green);
  font-weight: 500;
}

/* Why Recommended (ML Context) */
.product-reason {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--muted);
  background: rgba(76, 159, 63, 0.05);
  padding: 6px 10px;
  border-radius: 6px;
  margin-top: auto;
  margin-bottom: 12px;
}

.product-reason svg {
  flex-shrink: 0;
  color: var(--green);
}

/* Add to Cart Button */
.btn--add-cart {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px;
  background: var(--green);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn--add-cart:hover {
  background: var(--green-2);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(76, 159, 63, 0.3);
}

.btn--add-cart:active {
  transform: translateY(0);
}

.btn--add-cart svg {
  flex-shrink: 0;
}
```

---

### **4. Products Grid Layout**

```css
/* Products Grid */
.products-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

/* Responsive breakpoints */
@media (max-width: 1024px) {
  .products-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }
}

@media (max-width: 768px) {
  .products-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }
}

@media (max-width: 480px) {
  .products-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }
}
```

---

## 💻 JavaScript Integration (ML-Powered)

### **Complete Implementation**

#### File: `js/recommended-ml.js`

```javascript
// ============================================
// ML-Powered Recommendations Frontend
// ============================================

import { apiListProducts, apiCurrentUser } from "./api.js";
import { addToCart } from "./cart.js";
import { money, getFlashEffectivePrice } from "./utils.js";

// Configuration
const ML_API_URL = "http://localhost:5001";
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

// DOM Elements
const els = {
  status: document.getElementById("recStatus"),
  personal: document.getElementById("recPersonalGrid"),
  similar: document.getElementById("recSimilarGrid"),
  trending: document.getElementById("recTrendingGrid"),
  signal: document.getElementById("recSignal"),
  refresh: document.getElementById("recRefreshBtn"),
};

// Cache
const cache = new Map();

// ============================================
// 1. FETCH ML RECOMMENDATIONS
// ============================================

/**
 * Fetch recommendations from ML API
 */
async function fetchMLRecommendations(userId, options = {}) {
  const cacheKey = `ml_recs_${userId}`;

  // Check cache
  const cached = cache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    console.log("📦 Using cached recommendations");
    return cached.data;
  }

  try {
    const response = await fetch(`${ML_API_URL}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        n: options.n || 12,
        filter_purchased: options.filter_purchased ?? true,
        ...options,
      }),
      signal: AbortSignal.timeout(5000), // 5s timeout
    });

    if (!response.ok) {
      throw new Error(`ML API error: ${response.status}`);
    }

    const data = await response.json();

    // Cache the result
    cache.set(cacheKey, {
      data,
      timestamp: Date.now(),
    });

    return data;
  } catch (error) {
    console.warn("⚠️ ML API failed:", error);
    return null;
  }
}

/**
 * Fetch similar products from ML API
 */
async function fetchSimilarProducts(productId, n = 6) {
  try {
    const response = await fetch(`${ML_API_URL}/api/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_id: productId, n }),
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) throw new Error(`ML API error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.warn("⚠️ Similar products API failed:", error);
    return null;
  }
}

// ============================================
// 2. PRODUCT MAPPING & ENRICHMENT
// ============================================

/**
 * Map ML product IDs to VVV products
 * (Simplified: assumes 1:1 mapping or uses fallback)
 */
async function enrichRecommendations(mlRecommendations) {
  const products = await apiListProducts();
  const productMap = Object.fromEntries(products.map((p) => [String(p.id), p]));

  return mlRecommendations
    .map((rec) => {
      // Try to find matching product
      // In real implementation, use proper mapping table
      const product = productMap[String(rec.product_id)];

      if (product) {
        return {
          ...product,
          mlScore: rec.score,
          mlReason: getReasonText(rec.score),
        };
      }
      return null;
    })
    .filter(Boolean);
}

/**
 * Generate human-readable reason based on ML score
 */
function getReasonText(score) {
  if (score > 35) return "Bạn thường mua sản phẩm này";
  if (score > 25) return "Khách hàng thường mua cùng";
  if (score > 15) return "Phù hợp với sở thích của bạn";
  return "Sản phẩm được nhiều người yêu thích";
}

/**
 * Get badge type based on ML score (internal use only)
 */
function getBadgeType(score) {
  if (score > 35) return { type: "hot", text: "🔥 Bán chạy" };
  if (score > 25) return { type: "favorite", text: "💛 Yêu thích" };
  if (score > 15) return { type: "new", text: "✨ Gợi ý" };
  return null;
}

// ============================================
// 3. RENDER FUNCTIONS
// ============================================

/**
 * Render user interest chips
 */
function renderUserSignals(categories) {
  if (!els.signal) return;

  if (!categories || categories.length === 0) {
    els.signal.innerHTML =
      '<span class="chip chip--muted">🌱 Người dùng mới</span>';
    return;
  }

  const categoryEmojis = {
    veg: "🥬",
    fruit: "🍎",
    meat: "🥩",
    dry: "🌾",
    drink: "🥤",
    spice: "🌶️",
    household: "🧹",
    sweet: "🍰",
  };

  const chips = categories
    .slice(0, 5)
    .map((cat) => {
      const emoji = categoryEmojis[cat] || "📦";
      const name = cat.charAt(0).toUpperCase() + cat.slice(1);
      return `<span class="chip">${emoji} ${name}</span>`;
    })
    .join("");

  els.signal.innerHTML = chips;
}

/**
 * Render product cards
 */
function renderProducts(container, products, emptyText = "Không có sản phẩm") {
  if (!container) return;

  if (!products || products.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state__icon">📦</div>
        <p class="empty-state__text">${emptyText}</p>
      </div>
    `;
    return;
  }

  const cards = products
    .map((product) => {
      const effectivePrice = getFlashEffectivePrice(product);
      const hasDiscount = effectivePrice !== product.price;
      const badge = getBadgeType(product.mlScore || 0);

      return `
      <article class="product-card" data-id="${product.id}" data-ml-score="${
        product.mlScore || 0
      }">
        <!-- Image -->
        <div class="product-card__image">
          ${
            product.image
              ? `<img src="${product.image}" alt="${product.name}" loading="lazy" onerror="this.onerror=null; this.src='../images/brand/LogoVVV.png';" />`
              : `<div class="product-placeholder">${
                  product.emoji || "🛒"
                }</div>`
          }
          ${
            badge
              ? `<span class="product-badge product-badge--${badge.type}">${badge.text}</span>`
              : ""
          }
        </div>

        <!-- Body -->
        <div class="product-card__body">
          <h3 class="product-name">${product.name}</h3>
          
          <!-- Price -->
          <div class="product-price">
            <span class="price price--current">${money(effectivePrice)}</span>
            ${
              hasDiscount
                ? `<span class="price price--original">${money(
                    product.price
                  )}</span>
                 <span class="price-discount">-${Math.round(
                   (1 - effectivePrice / product.price) * 100
                 )}%</span>`
                : ""
            }
          </div>

          <!-- Meta -->
          <p class="product-meta">
            <span class="product-unit">${product.unit || "1 sản phẩm"}</span>
            <span class="product-stock">Còn hàng</span>
          </p>

          <!-- Reason (ML-powered, human-friendly) -->
          <div class="product-reason">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L4 12L1 9"></path>
            </svg>
            <span>${product.mlReason || "Được nhiều người yêu thích"}</span>
          </div>
        </div>

        <!-- Action -->
        <button class="btn btn--add-cart" data-action="add" data-id="${
          product.id
        }">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="9" cy="21" r="1"></circle>
            <circle cx="20" cy="21" r="1"></circle>
            <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
          </svg>
          Thêm vào giỏ
        </button>
      </article>
    `;
    })
    .join("");

  container.innerHTML = cards;

  // Stagger animation
  container.querySelectorAll(".product-card").forEach((card, i) => {
    card.style.animation = `fadeInUp 0.4s ease ${i * 0.05}s both`;
  });
}

/**
 * Update status message
 */
function updateStatus(message, type = "info") {
  if (!els.status) return;

  const icons = {
    loading: "⏳",
    success: "✓",
    error: "⚠️",
    info: "ℹ️",
  };

  els.status.innerHTML = `${icons[type] || ""} ${message}`;
  els.status.className = `section-description status-${type}`;
}

// ============================================
// 4. MAIN LOAD FUNCTION
// ============================================

async function loadRecommendations() {
  try {
    // Show loading
    updateStatus("Đang phân tích sở thích của bạn...", "loading");

    // Get current user
    const user = await apiCurrentUser();

    if (!user || !user.id) {
      // Guest user: show trending only
      await loadGuestRecommendations();
      return;
    }

    // Fetch ML recommendations
    const mlData = await fetchMLRecommendations(user.id, { n: 12 });

    if (!mlData) {
      // Fallback to rule-based
      console.log("Using fallback recommendations");
      await loadFallbackRecommendations(user);
      return;
    }

    // Enrich with product data
    const personalProducts = await enrichRecommendations(
      mlData.recommendations.slice(0, 8)
    );

    // Get similar products (based on top recommendation)
    let similarProducts = [];
    if (personalProducts.length > 0) {
      const topProductId = personalProducts[0].id;
      const similarData = await fetchSimilarProducts(topProductId, 6);
      if (similarData) {
        similarProducts = await enrichRecommendations(
          similarData.similar_items
        );
      }
    }

    // Get trending (use different weights)
    const trendingData = await fetchMLRecommendations(user.id, {
      n: 8,
      w_cf: 0.1,
      w_basket: 0.1,
      w_pop: 0.8,
    });
    const trendingProducts = trendingData
      ? await enrichRecommendations(trendingData.recommendations)
      : [];

    // Extract user interests for chips
    const userCategories = personalProducts
      .map((p) => p.cat)
      .filter((cat, i, arr) => arr.indexOf(cat) === i)
      .slice(0, 5);

    // Render everything
    renderUserSignals(userCategories);
    renderProducts(els.personal, personalProducts, "Chưa có gợi ý cá nhân");
    renderProducts(els.similar, similarProducts, "Chưa có sản phẩm tương tự");
    renderProducts(els.trending, trendingProducts, "Đang tải...");

    // Update status
    updateStatus(
      `${personalProducts.length} sản phẩm được chọn lọc dựa trên lịch sử mua hàng của bạn`,
      "success"
    );

    console.log("✅ ML recommendations loaded successfully");
  } catch (error) {
    console.error("❌ Load recommendations error:", error);
    updateStatus("Đã xảy ra lỗi khi tải gợi ý", "error");
  }
}

// ============================================
// 5. FALLBACK FUNCTIONS
// ============================================

async function loadGuestRecommendations() {
  const products = await apiListProducts();
  const trending = products
    .sort((a, b) => (b.pop || 0) - (a.pop || 0))
    .slice(0, 12)
    .map((p) => ({ ...p, mlReason: "Sản phẩm phổ biến" }));

  renderUserSignals([]);
  renderProducts(els.personal, trending.slice(0, 8));
  renderProducts(els.similar, []);
  renderProducts(els.trending, trending.slice(8, 12));

  updateStatus("Đăng nhập để nhận gợi ý cá nhân hóa", "info");
}

async function loadFallbackRecommendations(user) {
  // Use existing rule-based logic from recommended.js
  // This is your backup when ML API is down
  console.log("Loading fallback recommendations...");

  const products = await apiListProducts();
  const trending = products
    .sort((a, b) => (b.pop || 0) - (a.pop || 0))
    .slice(0, 12);

  renderProducts(els.personal, trending.slice(0, 8));
  renderProducts(els.similar, trending.slice(4, 10));
  renderProducts(els.trending, trending.slice(0, 8));

  updateStatus("Hiển thị gợi ý mặc định (ML API không khả dụng)", "info");
}

// ============================================
// 6. EVENT HANDLERS
// ============================================

function bindEvents() {
  // Refresh button
  if (els.refresh) {
    els.refresh.addEventListener("click", () => {
      cache.clear();
      loadRecommendations();
    });
  }

  // Add to cart
  document.addEventListener("click", (e) => {
    const btn = e.target.closest('[data-action="add"]');
    if (!btn) return;

    e.preventDefault();
    const productId = btn.dataset.id;
    if (!productId) return;

    addToCart(productId, 1);
    showToast("✅ Đã thêm vào giỏ hàng");

    // Track ML event (for analytics)
    trackMLEvent("add_to_cart", {
      product_id: productId,
      source: "ml_recommendation",
    });
  });
}

// ============================================
// 7. UTILITIES
// ============================================

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast toast--success";
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.classList.add("show"), 10);
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

function trackMLEvent(eventName, data) {
  // Send to analytics
  console.log("📊 Track ML event:", eventName, data);

  // Store locally for admin dashboard
  const events = JSON.parse(localStorage.getItem("ml_events") || "[]");
  events.push({
    event: eventName,
    timestamp: new Date().toISOString(),
    ...data,
  });
  localStorage.setItem("ml_events", JSON.stringify(events.slice(-1000)));
}

// ============================================
// 8. INITIALIZATION
// ============================================

(function init() {
  bindEvents();

  if (document.readyState !== "loading") {
    loadRecommendations();
  } else {
    document.addEventListener("DOMContentLoaded", loadRecommendations);
  }
})();

export { loadRecommendations, fetchMLRecommendations };
```

---

## 🎬 Animation & Transitions

### **CSS Animations**

```css
/* Fade In Up Animation */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Slide In Right (for toast) */
@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(100px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* Pulse Animation (for loading) */
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

/* Shimmer Effect (loading skeleton) */
@keyframes shimmer {
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
}
```

---

## ⏳ Loading States

### **Loading Skeleton**

```html
<!-- Loading skeleton for product cards -->
<div class="product-card product-card--loading">
  <div class="skeleton skeleton--image"></div>
  <div class="skeleton skeleton--text skeleton--title"></div>
  <div class="skeleton skeleton--text skeleton--price"></div>
  <div class="skeleton skeleton--button"></div>
</div>
```

```css
/* Loading Skeleton */
.product-card--loading {
  pointer-events: none;
}

.skeleton {
  background: linear-gradient(90deg, #f0f0f0 0%, #f8f8f8 50%, #f0f0f0 100%);
  background-size: 1000px 100%;
  animation: shimmer 2s infinite linear;
  border-radius: 8px;
}

.skeleton--image {
  width: 100%;
  height: 200px;
  margin-bottom: 16px;
}

.skeleton--text {
  height: 16px;
  margin-bottom: 12px;
}

.skeleton--title {
  width: 80%;
}

.skeleton--price {
  width: 60%;
}

.skeleton--button {
  width: 100%;
  height: 44px;
  margin-top: 16px;
}
```

### **Loading State Function**

```javascript
function showLoadingState(container) {
  if (!container) return;

  const skeletons = Array.from(
    { length: 8 },
    () => `
    <div class="product-card product-card--loading">
      <div class="skeleton skeleton--image"></div>
      <div class="product-card__body">
        <div class="skeleton skeleton--text skeleton--title"></div>
        <div class="skeleton skeleton--text skeleton--price"></div>
        <div class="skeleton skeleton--button"></div>
      </div>
    </div>
  `
  ).join("");

  container.innerHTML = skeletons;
}

// Usage:
showLoadingState(els.personal);
```

---

## 📱 Responsive Design

### **Mobile Optimizations**

```css
/* Mobile: < 768px */
@media (max-width: 768px) {
  .page-hero {
    padding: 24px 0;
  }

  .page-title {
    font-size: 28px;
  }

  .section-card {
    padding: 20px 16px;
    border-radius: 12px;
  }

  .section-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .btn--refresh {
    width: 100%;
    justify-content: center;
  }

  .section-title {
    font-size: 22px;
  }

  .product-card__image {
    height: 160px;
  }

  .product-name {
    font-size: 14px;
  }

  .price--current {
    font-size: 16px;
  }
}

/* Tablet: 768px - 1024px */
@media (min-width: 768px) and (max-width: 1024px) {
  .products-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }

  .section-card {
    padding: 28px;
  }
}
```

---

## 🎯 Empty States

```html
<div class="empty-state">
  <div class="empty-state__icon">📦</div>
  <h3 class="empty-state__title">Chưa có gợi ý</h3>
  <p class="empty-state__text">
    Hãy bắt đầu mua sắm để nhận được gợi ý cá nhân hóa!
  </p>
  <a href="index.html" class="btn btn--primary"> Khám phá sản phẩm </a>
</div>
```

```css
.empty-state {
  text-align: center;
  padding: 60px 20px;
  grid-column: 1 / -1;
}

.empty-state__icon {
  font-size: 64px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state__title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 8px;
}

.empty-state__text {
  font-size: 14px;
  color: var(--muted);
  margin-bottom: 24px;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}
```

---

## 🎨 Toast Notifications

```css
/* Toast Container */
.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: white;
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 14px;
  font-weight: 500;
  z-index: 9999;
  opacity: 0;
  transform: translateX(100px);
  transition: all 0.3s ease;
}

.toast.show {
  opacity: 1;
  transform: translateX(0);
}

.toast--success {
  border-left: 4px solid var(--green);
}

.toast--error {
  border-left: 4px solid #ff6b6b;
}

@media (max-width: 768px) {
  .toast {
    bottom: 16px;
    right: 16px;
    left: 16px;
    font-size: 13px;
  }
}
```

---

## ✅ Checklist Triển Khai

### **Phase 1: Setup (1 ngày)**

- [ ] Copy CSS từ guide vào `css/style.css`
- [ ] Tạo file `js/recommended-ml.js`
- [ ] Update `html/recommended.html` với HTML structure mới
- [ ] Test ML API connection

### **Phase 2: Integration (2 ngày)**

- [ ] Implement product mapping logic
- [ ] Connect ML API endpoints
- [ ] Add loading states
- [ ] Implement fallback logic

### **Phase 3: UI Polish (1 ngày)**

- [ ] Add animations
- [ ] Optimize responsive design
- [ ] Add empty states
- [ ] Test across devices

### **Phase 4: Testing (1 ngày)**

- [ ] Test với users có lịch sử mua hàng
- [ ] Test với guest users
- [ ] Test khi ML API down
- [ ] Performance testing

---

## 🎓 Best Practices Summary

### **DO's ✅**

- Giữ UI đơn giản, không hiển thị số liệu kỹ thuật
- Giải thích TẠI SAO gợi ý (human-friendly reasons)
- Smooth animations và transitions
- Loading states cho UX tốt
- Fallback khi ML API fails
- Cache recommendations để tăng tốc

### **DON'Ts ❌**

- Không hiển thị ML scores (38.5, 42.1, etc.)
- Không dùng thuật ngữ kỹ thuật ("NMF", "Collaborative Filtering")
- Không overload UI với quá nhiều thông tin
- Không bỏ qua loading states
- Không hard-code values

---

## 📊 Success Metrics

### **Track These KPIs:**

1. **Click-through Rate:** Users click vào recommendations
2. **Add-to-cart Rate:** Users thêm sản phẩm vào giỏ
3. **Conversion Rate:** Từ recommendation → purchase
4. **Time on Page:** Thời gian ở trang recommendations
5. **Bounce Rate:** Users rời đi ngay

### **Target Values:**

- CTR: > 12%
- Add-to-cart: > 8%
- Conversion: > 5%
- Avg time: > 2 minutes

---

**Prepared by:** GitHub Copilot  
**Version:** 1.0  
**Last Updated:** 13 January 2026
