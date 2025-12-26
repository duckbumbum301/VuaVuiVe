// js/season.js — Gợi ý mua sắm theo mùa
import { apiListProducts } from "./api.js";
import { addToCart } from "./cart.js";
import { money, getFlashEffectivePrice } from "./utils.js";

const els = {
  grid: document.getElementById("seasonGrid"),
  combos: document.getElementById("seasonCombos"),
  comboStatus: document.getElementById("comboStatus"),
  title: document.getElementById("seasonTitle"),
  desc: document.getElementById("seasonDesc"),
  eyebrow: document.getElementById("seasonEyebrow"),
  tabs: document.getElementById("seasonTabs"),
  heroIcon: document.querySelector(".illus__emoji"),
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

const SEASONS = {
  spring: {
    label: "Xuân",
    emoji: "🌱",
    title: "Tươi mới, thanh nhẹ",
    desc: "Rau xanh, trái cây mọng nước và trà thảo mộc giúp thanh lọc, nhẹ bụng.",
    match: (p) =>
      ["veg", "fruit", "drink"].includes(p.cat) &&
      ["leaf", "herb", "mushroom", "fresh", "tea", "juice"].includes(p.sub),
  },
  summer: {
    label: "Hè",
    emoji: "🌊",
    title: "Giải nhiệt, chống nắng",
    desc: "Nước ép, trái cây theo mùa, sữa chua và đồ uống mát lạnh.",
    match: (p) =>
      (p.cat === "drink" && ["juice", "tea", "can", "milk"].includes(p.sub)) ||
      (p.cat === "fruit" && ["seasonal", "fresh", "dried"].includes(p.sub)),
  },
  autumn: {
    label: "Thu",
    emoji: "🍂",
    title: "Bổ sung năng lượng",
    desc: "Ngũ cốc, hạt, đồ khô và trái cây ngọt dịu cho tiết trời mát.",
    match: (p) =>
      p.cat === "dry" ||
      p.cat === "sweet" ||
      (p.cat === "fruit" && ["dried", "mixed", "gift"].includes(p.sub)),
  },
  winter: {
    label: "Đông",
    emoji: "🔥",
    title: "Ấm bụng, giàu đạm",
    desc: "Thịt cá, hải sản, gia vị nấu lẩu và mì, kèm đồ uống nóng.",
    match: (p) =>
      p.cat === "meat" ||
      p.cat === "spice" ||
      (p.cat === "dry" && ["noodle", "rice"].includes(p.sub)) ||
      (p.cat === "drink" && ["coffee", "tea"].includes(p.sub)),
  },
};

let productsCache = [];
let productMap = new Map();

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

function renderCards(
  target,
  products,
  emptyText = "Chưa có gợi ý cho mùa này."
) {
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

function selectSeason(key) {
  const season = SEASONS[key];
  if (!season) return;
  if (els.title) els.title.textContent = season.title;
  if (els.desc) els.desc.textContent = season.desc;
  if (els.eyebrow) els.eyebrow.textContent = season.label;
  if (els.heroIcon) els.heroIcon.textContent = season.emoji;

  // Tab active state
  els.tabs?.querySelectorAll("[data-season]").forEach((btn) => {
    const active = btn.dataset.season === key;
    btn.classList.toggle("chip--active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });

  const list = productsCache
    .filter((p) => season.match(p))
    .sort((a, b) => (b.pop || 0) - (a.pop || 0))
    .slice(0, 12);
  renderCards(els.grid, list, "Đang tải sản phẩm...");
}

async function loadProducts() {
  if (productsCache.length) return productsCache;
  productsCache = await apiListProducts();
  productMap = new Map(productsCache.map((p) => [String(p.id), p]));
  return productsCache;
}

async function fetchSeasonCombos() {
  try {
    const res = await fetch("http://localhost:3000/api/recommend/seasonal", {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data;
  } catch (err) {
    console.error("Failed to fetch seasonal combos", err);
    return null;
  }
}

function renderCombos(data) {
  if (!els.combos) return;
  if (!data || !Array.isArray(data.rules) || data.rules.length === 0) {
    els.combos.innerHTML = `<p class="muted">Chưa có combo đủ dữ liệu cho mùa này.</p>`;
    if (els.comboStatus) {
      els.comboStatus.textContent = "Dùng gợi ý phổ biến làm mặc định.";
    }
    return;
  }

  if (els.comboStatus) {
    els.comboStatus.textContent = `Mùa hiện tại: ${data.season}. Dựa trên luật kết hợp trong đơn hàng mùa này.`;
  }

  const cards = data.rules.slice(0, 6).map((rule) => {
    const a = productMap.get(String(rule.triggerId)) || {};
    const b = productMap.get(String(rule.recommendId)) || {};

    const cardTitle = `${rule.trigger} + ${rule.recommend}`;
    const confPct = Math.round((rule.confidence || 0) * 100);

    const thumbA = a.image
      ? `<img src="${a.image}" alt="${rule.trigger}" loading="lazy" decoding="async" onerror="this.onerror=null; this.src='../images/brand/LogoVVV.png';" />`
      : "🛒";
    const thumbB = b.image
      ? `<img src="${b.image}" alt="${rule.recommend}" loading="lazy" decoding="async" onerror="this.onerror=null; this.src='../images/brand/LogoVVV.png';" />`
      : "🛒";

    return `
      <article class="card card--combo">
        <div class="combo-thumb">
          <div class="combo-thumb__item">${thumbA}</div>
          <span class="combo-plus">+</span>
          <div class="combo-thumb__item">${thumbB}</div>
        </div>
        <div class="name">${cardTitle}</div>
        <p class="muted">Xác suất mua kèm ~${confPct}% (support ${
      rule.support ?? 0
    })</p>
        <div class="card__foot">
          <button class="btn btn--cart" data-action="add" data-id="${
            rule.triggerId
          }">Thêm ${rule.trigger}</button>
          <button class="btn btn--cart" data-action="add" data-id="${
            rule.recommendId
          }">Thêm ${rule.recommend}</button>
        </div>
      </article>
    `;
  });

  els.combos.innerHTML = cards.join("");
}

function bindTabs() {
  els.tabs?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-season]");
    if (!btn) return;
    selectSeason(btn.dataset.season);
  });
}

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

(async function init() {
  bindTabs();
  bindAddToCart();
  try {
    await loadProducts();
    selectSeason("spring");

    const comboData = await fetchSeasonCombos();
    renderCombos(comboData);
  } catch (err) {
    console.error("Cannot load seasonal suggestions", err);
    renderCards(els.grid, [], "Không tải được dữ liệu.");
    renderCombos(null);
  }
})();
