// ============================================
// ML-Powered Recommendations Frontend
// ============================================

import { apiListProducts, apiCurrentUser } from "./api.js";
import { addToCart } from "./cart.js";
import { money } from "./utils.js";

// Get flash price function
function getFlashEffectivePrice(product) {
  // Fallback if no price
  if (!product || !product.price) {
    return 0;
  }

  try {
    // Try to get flash price if available
    const now = new Date();
    const h = now.getHours();

    if (product.flashSlot && product.flashDiscount) {
      const isMorning = product.flashSlot === "morning" && h >= 6 && h < 8;
      const isAfternoon =
        product.flashSlot === "afternoon" && h >= 16 && h < 18;

      if (isMorning || isAfternoon) {
        return Math.round(product.price * (1 - product.flashDiscount));
      }
    }
  } catch (err) {
    console.warn("Flash price calculation error:", err);
  }

  return product.price;
}

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
  const cacheVariant = [
    options.n ?? 12,
    options.filter_purchased ?? true,
    options.user_phone ?? "",
    options.recency_decay_days ?? "",
    options.w_cf ?? "",
    options.w_basket ?? "",
    options.w_pop ?? "",
    options.model_user_id ?? "",
  ].join("|");
  const cacheKey = `ml_recs_${userId}_${cacheVariant}`;

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
        user_email: options.user_email,
        user_name: options.user_name,
        user_phone: options.user_phone,
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
 * NEW: ML API đã trả về full VVV product data
 */
async function enrichRecommendations(mlRecommendations) {
  if (!mlRecommendations || !Array.isArray(mlRecommendations)) {
    console.warn("Invalid mlRecommendations:", mlRecommendations);
    return [];
  }

  // NEW APPROACH: ML API (với adapter) đã trả về VVV products với đủ thông tin
  // Chỉ cần enrich thêm flash sale price nếu có

  return mlRecommendations
    .filter((rec) => rec && rec.product_id && rec.name && rec.price) // Filter invalid items
    .map((rec) => {
      // rec đã có: product_id, score, name, price, image, category, reason
      const product = {
        id: rec.product_id,
        price: rec.price,
        flashSlot: rec.flashSlot,
        flashDiscount: rec.flashDiscount,
      };

      const flashPrice = getFlashEffectivePrice(product);

      return {
        id: rec.product_id,
        name: rec.name,
        price: rec.price,
        image: rec.image,
        cat: rec.category,
        unit: rec.unit || "1 sản phẩm",
        mlScore: rec.score || 0,
        mlReason: rec.reason || "Gợi ý cho bạn", // Đã có từ API
        flashPrice: flashPrice !== rec.price ? flashPrice : null,
      };
    });
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
    .filter((cat) => cat && typeof cat === "string") // Filter out undefined/null
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
    .filter((product) => product && product.price) // Filter products without price
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

/**
 * Show loading skeleton
 */
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

// ============================================
// 4. MAIN LOAD FUNCTION
// ============================================

async function loadRecommendations() {
  try {
    // Show loading
    updateStatus("Đang phân tích sở thích của bạn...", "loading");
    showLoadingState(els.personal);
    showLoadingState(els.similar);
    showLoadingState(els.trending);

    // Get current user
    const user = await apiCurrentUser();

    if (!user || !user.id) {
      // Guest user: show trending only
      await loadGuestRecommendations();
      return;
    }

    // Fetch ML recommendations
    const mlData = await fetchMLRecommendations(user.id, {
      n: 12,
      user_email: user.email,
      user_name: user.name,
      user_phone: user.phone,
      // This section is "Tiếp nối" (continue what you like): allow repeat buys.
      filter_purchased: false,
      // Favor the most recent orders strongly.
      recency_decay_days: 7,
    });

    console.log(" ML Data received:", mlData);

    if (!mlData || !mlData.recommendations) {
      // Fallback to rule-based
      console.log("Using fallback recommendations");
      await loadFallbackRecommendations(user);
      return;
    }

    // Enrich with product data
    const personalProducts = await enrichRecommendations(
      mlData.recommendations.slice(0, 8)
    );

    console.log(" Personal products enriched:", personalProducts);

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
      user_email: user.email,
      user_name: user.name,
      user_phone: user.phone,
    });
    const trendingProducts = trendingData
      ? await enrichRecommendations(trendingData.recommendations)
      : [];

    // Extract user interests for chips
    const userCategories = personalProducts
      .filter((p) => p && p.cat) // Filter out products without category
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

    console.log(" ML recommendations loaded successfully");
  } catch (error) {
    console.error(" Load recommendations error:", error);
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
  // Use existing rule-based logic when ML API is down
  console.log("Loading fallback recommendations...");

  const products = await apiListProducts();
  const trending = products
    .sort((a, b) => (b.pop || 0) - (a.pop || 0))
    .slice(0, 12)
    .map((p) => ({ ...p, mlReason: "Sản phẩm được nhiều người yêu thích" }));

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
    showToast(" Đã thêm vào giỏ hàng");

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
  console.log(" Track ML event:", eventName, data);

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
  // Check if required DOM elements exist
  const requiredElements = [
    "recPersonalGrid",
    "recSimilarGrid",
    "recTrendingGrid",
    "recStatus",
  ];
  const missing = requiredElements.filter((id) => !document.getElementById(id));

  if (missing.length > 0) {
    console.error("❌ Missing required elements:", missing);
    return;
  }

  bindEvents();

  if (document.readyState !== "loading") {
    loadRecommendations();
  } else {
    document.addEventListener("DOMContentLoaded", loadRecommendations);
  }
})();

export { loadRecommendations, fetchMLRecommendations };
