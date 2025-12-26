// js/recommended.js — Gợi ý cá nhân dựa trên lịch sử mua hàng
import { apiListProducts, apiListOrders, apiCurrentUser } from "./api.js";
import { addToCart } from "./cart.js";
import { money, getFlashEffectivePrice } from "./utils.js";

// API Base URL
const API_BASE =
  window.location.origin.includes("localhost") ||
  window.location.origin.includes("127.0.0.1")
    ? "http://localhost:3000"
    : window.location.origin;

const els = {
  status: document.getElementById("recStatus"),
  personal: document.getElementById("recPersonalGrid"),
  similar: document.getElementById("recSimilarGrid"),
  trending: document.getElementById("recTrendingGrid"),
  signal: document.getElementById("recSignal"),
  refresh: document.getElementById("recRefreshBtn"),
};

const CAT_THUMB = {
  veg: "thumb--veg",
  fruit: "thumb--fruit",
  meat: "thumb--meat",
  dry: "thumb--dry",
  drink: "thumb--drink",
  spice: "thumb--spice",
  household: "thumb--household",
  sweet: "thumb--sweet",
};

function ensureToast() {
  let c = document.getElementById("toastContainer");
  if (!c) {
    c = document.createElement("div");
    c.id = "toastContainer";
    c.className = "toast-container";
    document.body.appendChild(c);
  }
  return c;
}
function showToast(msg) {
  const c = ensureToast();
  const t = document.createElement("div");
  t.className = "toast";
  t.innerHTML = `<span class="toast-message">${msg}</span><button class="toast-close" aria-label="Close">×</button>`;
  c.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  const remove = () => {
    t.classList.remove("show");
    t.addEventListener("transitionend", () => t.remove(), { once: true });
  };
  const timer = setTimeout(remove, 2600);
  t.querySelector(".toast-close")?.addEventListener("click", () => {
    clearTimeout(timer);
    remove();
  });
}

function renderCards(target, products, emptyText = "Không có gợi ý phù hợp.") {
  if (!target) return;
  if (!products || !products.length) {
    target.innerHTML = `<p class="muted">${emptyText}</p>`;
    return;
  }
  const cards = products.map((p) => {
    const thumb = p.image
      ? `<img src="${p.image}" alt="${p.name}" loading="lazy" decoding="async" onerror="this.onerror=null; this.src='../images/brand/LogoVVV.png';" style="width:100%;height:100%;object-fit:contain;" />`
      : p.emoji || "🛒";
    const catClass = CAT_THUMB[p.cat] || "thumb--veg";
    const eff = getFlashEffectivePrice(p);
    const isDiscount = eff !== p.price;
    const priceHtml = isDiscount
      ? `<span class="price price--sale">${money(
          eff
        )}</span> <span class="price price--orig">${money(p.price)}</span>`
      : `<span class="price">${money(p.price)}</span>`;
    return `
      <article class="card" data-id="${p.id}">
        <div class="thumb ${catClass}" aria-hidden="true">${thumb}</div>
        <div class="name">${p.name}</div>
        <div class="meta">
          <div class="pricegroup">${priceHtml}</div>
          <span class="muted">${p.unit || ""}</span>
        </div>
        <div class="card__foot">
          <button class="btn btn--cart" data-action="add" data-id="${
            p.id
          }">Thêm vào giỏ</button>
        </div>
      </article>`;
  });
  target.innerHTML = cards.join("");
}

/**
 * Fetch recommendations từ backend API
 */
async function fetchRecommendations() {
  const user = await apiCurrentUser();

  if (!user || !user.id) {
    console.warn("No user logged in");
    return null;
  }

  try {
    const response = await fetch(
      `${API_BASE}/api/recommendations?userId=${user.id}`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch recommendations:", error);
    return null;
  }
}

/**
 * Render user signals (top categories chips)
 */
function renderUserSignals(signals) {
  if (!els.signal) return;

  if (!signals.hasHistory) {
    els.signal.innerHTML =
      '<span class="chip chip--muted">Người dùng mới</span>';
    return;
  }

  const chips = signals.topCategories
    .map((cat) => `<span class="chip">${cat}</span>`)
    .join("");

  els.signal.innerHTML = chips;
}

/**
 * Update status message
 */
function updateStatus(message, type = "info") {
  if (!els.status) return;
  els.status.textContent = message;
  els.status.className = `muted status-${type}`;
}

async function loadRecommendations() {
  try {
    updateStatus("Đang phân tích đơn hàng...", "loading");

    const data = await fetchRecommendations();

    if (!data) {
      // Fallback to old implementation
      console.log("Using fallback recommendation system");
      await loadRecommendationsFallback();
      return;
    }

    // Render user signals
    renderUserSignals(data.userSignals);

    // Render grids
    renderCards(els.personal, data.personal, "Chưa có gợi ý cá nhân.");
    renderCards(els.similar, data.similar, "Chưa có sản phẩm tương tự.");
    renderCards(els.trending, data.trending, "Đang tải...");

    // Update status
    const statusMsg = data.userSignals.hasHistory
      ? `Phân tích ${data.userSignals.purchaseCount} đơn hàng • ${data.personal.length} gợi ý cá nhân`
      : "Bắt đầu mua sắm để nhận gợi ý cá nhân hóa!";

    updateStatus(statusMsg, "success");

    console.log("✅ Recommendations rendered:", data.metadata);
  } catch (error) {
    console.error("Render error:", error);
    updateStatus("Đã xảy ra lỗi khi tải gợi ý.", "error");
  }
}

/**
 * Fallback implementation (old logic)
 */
async function loadRecommendationsFallback() {
  try {
    if (els.status) els.status.textContent = "Đang tải dữ liệu...";
    const [products, orders, user] = await Promise.all([
      apiListProducts(),
      apiListOrders(),
      apiCurrentUser(),
    ]);
    const map = Object.fromEntries(products.map((p) => [String(p.id), p]));
    const mine = filterOrdersByUser(orders, user);
    const history = buildHistory(mine, map);
    updateSignals(history);

    if (!user) {
      if (els.status)
        els.status.textContent =
          "Vui lòng đăng nhập để nhận gợi ý chuẩn hơn. Dưới đây là gợi ý mặc định.";
    } else if (!mine.length) {
      if (els.status)
        els.status.textContent =
          "Chưa có đơn hàng, bạn hãy thử mua để cá nhân hóa gợi ý.";
    } else if (els.status) {
      els.status.textContent = `Gợi ý sản phẩm dựa trên các danh mục bạn thường xuyên mua sắm.`;
    }

    // 1) Mua lại nhanh
    const reorder = pickReorder(history.byProduct, map, products, 8);
    renderCards(els.personal, reorder, "Chưa có đơn nào để gợi ý.");

    // 2) Sản phẩm tương tự
    const similar = pickSimilar(products, history, map, 12);
    renderCards(els.similar, similar, "Sẽ hiển thị khi bạn có lịch sử mua.");

    // 3) Xu hướng
    const trending = pickTrending(products, 12);
    renderCards(els.trending, trending, "Đang tải sản phẩm...");
  } catch (err) {
    console.error("Load recommendations failed", err);
    if (els.status)
      els.status.textContent = "Không tải được dữ liệu, vui lòng thử lại.";
    renderCards(els.personal, [], "Không có dữ liệu.");
    renderCards(els.similar, [], "Không có dữ liệu.");
    renderCards(els.trending, [], "Không có dữ liệu.");
  }
}

function filterOrdersByUser(list, user) {
  if (!user) return [];
  const email = (user.email || "").toLowerCase();
  const phone = user.phone || "";
  return (list || []).filter((o) => {
    const ue = (o.user?.email || "").toLowerCase();
    const up = o.user?.phone || "";
    return (email && ue && ue === email) || (phone && up && up === phone);
  });
}

function buildHistory(orders, productMap) {
  const byProduct = new Map();
  const byCat = new Map();
  const bySub = new Map();

  (orders || []).forEach((o) => {
    Object.entries(o.items || {}).forEach(([pid, qty]) => {
      const q = Number(qty) || 0;
      if (q <= 0) return;
      const key = String(pid);
      byProduct.set(key, (byProduct.get(key) || 0) + q);
      const p = productMap[key];
      if (p) {
        byCat.set(p.cat, (byCat.get(p.cat) || 0) + q);
        bySub.set(p.sub, (bySub.get(p.sub) || 0) + q);
      }
    });
  });
  return { byProduct, byCat, bySub };
}

function topKeys(map, limit = 3) {
  return Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k]) => k);
}

function pickReorder(byProduct, productMap, products, limit = 8) {
  const ids = Array.from(byProduct.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => id);
  const list = ids
    .map((id) => productMap[id])
    .filter(Boolean)
    .slice(0, limit);
  // Fallback: nếu không đủ, bù thêm sản phẩm phổ biến
  if (list.length < limit) {
    const missing = products
      .filter((p) => !ids.includes(String(p.id)))
      .sort((a, b) => (b.pop || 0) - (a.pop || 0))
      .slice(0, limit - list.length);
    return list.concat(missing);
  }
  return list;
}

function pickSimilar(products, history, productMap, limit = 12) {
  const purchased = new Set(Array.from(history.byProduct.keys()));
  const topCats = new Set(topKeys(history.byCat, 3));
  const topSubs = new Set(topKeys(history.bySub, 4));

  const scored = products
    .filter((p) => !purchased.has(String(p.id)))
    .map((p) => {
      let score = 0;
      if (topCats.has(p.cat)) score += 2;
      if (topSubs.has(p.sub)) score += 3;
      score += Number(p.pop || 0) / 50;
      return { p, score };
    })
    .filter((item) => item.score > 0.1)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.p);

  // Fallback nếu quá ít
  if (scored.length < limit) {
    const extra = products
      .filter((p) => !purchased.has(String(p.id)))
      .sort((a, b) => (b.pop || 0) - (a.pop || 0))
      .slice(0, limit - scored.length);
    return scored.concat(extra);
  }
  return scored;
}

function pickTrending(products, limit = 12) {
  return products
    .slice()
    .sort((a, b) => (b.pop || 0) - (a.pop || 0))
    .slice(0, limit);
}

function updateSignals(history) {
  if (!els.signal) return;
  const cats = topKeys(history.byCat, 3);
  const subs = topKeys(history.bySub, 3);
  const chips = [];
  if (cats.length)
    chips.push(...cats.map((c) => `<span class="chip">${c}</span>`));
  if (subs.length)
    chips.push(...subs.map((s) => `<span class="chip chip--soft">${s}</span>`));
  els.signal.innerHTML =
    chips.join("") ||
    '<span class="muted">Chưa có dữ liệu, hãy đăng nhập và mua thử một sản phẩm.</span>';
}

// Sự kiện thêm vào giỏ
function bindAddToCart() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest('[data-action="add"]');
    if (!btn) return;
    e.preventDefault();
    const pid = btn.dataset.id || btn.closest(".card")?.dataset.id;
    if (!pid) return;
    addToCart(pid, 1);
    showToast("Đã thêm vào giỏ hàng.");
    document.dispatchEvent(new Event("cart:changed"));
  });
}

(function init() {
  if (els.refresh) els.refresh.addEventListener("click", loadRecommendations);
  bindAddToCart();
  if (document.readyState !== "loading") loadRecommendations();
  else document.addEventListener("DOMContentLoaded", loadRecommendations);
})();
