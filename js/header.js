// js/header.js — Inject a consistent site header across pages
import { bindMegaMenu } from "./menu.js";
import { getCart, loadCart } from "./cart.js";
import { apiListProducts, apiCurrentUser } from "./api.js";

function sumCartQty() {
  try {
    const entries = Object.entries(getCart()).filter(([, q]) => q > 0);
    return entries.reduce((s, [, q]) => s + q, 0);
  } catch {
    return 0;
  }
}

function buildHeaderHTML() {
  const iconSearch = `
    <svg class="searchbox__icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2" />
      <line x1="16.65" y1="16.65" x2="21" y2="21" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
    </svg>`;

  // Trang chủ mong muốn: Home/Hôm
  const homeHref = "../Home/H%C3%B4m/index.html";
  const isHome = /\/html\/index\.html$/i.test(location.pathname);
  const catalogHref = isHome ? "#catalog" : "/html/index.html#catalog";
  const makeProductHref = (id) => {
    const pid = String(id);
    return isHome
      ? `?product=${pid}#catalog`
      : `/html/index.html?product=${pid}#catalog`;
  };

  return `
    <div class="container header__row">
      <a href="${homeHref}" class="logo" aria-label="Trang chủ Vựa Vui Vẻ"><img src="../images/brand/LogoVVV.png" alt="Vựa Vui Vẻ" class="logo__img" /></a>
      <nav class="nav" aria-label="Điều hướng chính">
        <div class="nav-item nav-item--dropdown" aria-expanded="false">
          <button
            class="nav-link nav-link--dropdown"
            id="productsMenuToggle"
            aria-expanded="false"
            aria-haspopup="true"
            aria-controls="productsMegaMenu"
            aria-label="Mở menu Sản phẩm"
            type="button"
          >
            Sản phẩm
            <svg
              class="nav-arrow"
              width="12"
              height="8"
              viewBox="0 0 12 8"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M1 1.5L6 6.5L11 1.5"
                stroke="currentColor"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </button>
          <div
            id="productsMegaMenu"
            class="mega-menu"
            role="menu"
            aria-labelledby="productsMenuToggle"
            hidden
          >
            <div class="mega-menu__content">
              <!-- Rau củ -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Rau Củ</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="veg"
                      data-sub="leaf"
                      >Rau lá</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="veg"
                      data-sub="root"
                      >Củ & rễ</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="veg"
                      data-sub="cabbage"
                      >Họ cải</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="veg"
                      data-sub="mushroom"
                      >Nấm Các Loại</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="veg"
                      data-sub="herb"
                      >Rau thơm</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Trái cây -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Trái Cây</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="fruit"
                      data-sub="fresh"
                      >Trái cây tươi</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="fruit"
                      data-sub="imported"
                      >Trái cây nhập khẩu</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="fruit"
                      data-sub="seasonal"
                      >Trái cây theo mùa</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="fruit"
                      data-sub="dried"
                      >Trái cây sấy & đóng gói</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="fruit"
                      data-sub="gift"
                      >Giỏ quà - Combo trái cây</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Thịt - Cá -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Thịt - Cá</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="meat"
                      data-sub="pork"
                      >Thịt heo</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="meat"
                      data-sub="redmeat"
                      >Thịt bò & Thịt đỏ</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="meat"
                      data-sub="poultry"
                      >Gia cầm (Gà, Vịt)</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="meat"
                      data-sub="fish"
                      >Cá tươi</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="meat"
                      data-sub="seafood"
                      >Hải sản</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Bia, nước giải khát -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Nước Giải Khát</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="drink"
                      data-sub="can"
                      >Nước đóng chai</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="drink"
                      data-sub="juice"
                      >Nước ép & sinh tố</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="drink"
                      data-sub="milk"
                      >Sữa & chế phẩm sữa</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="drink"
                      data-sub="tea"
                      >Trà & thảo mộc</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="drink"
                      data-sub="coffee"
                      >Cà phê</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Gạo, bột, đồ khô -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Đồ Khô</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="dry"
                      data-sub="rice"
                      >Gạo & Ngũ cốc</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="dry"
                      data-sub="noodle"
                      >Mì / Bún / Phở khô</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="dry"
                      data-sub="beans-nuts"
                      >Đậu & Hạt</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="dry"
                      data-sub="flour-mix"
                      >Bột & Hỗn hợp nấu ăn</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="dry"
                      data-sub="canned"
                      >Đồ hộp & Đóng gói</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Gia vị -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Gia Vị</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="spice"
                      data-sub="oil"
                      >Dầu ăn & Mỡ</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="spice"
                      data-sub="sauce"
                      >Nước chấm & Sốt</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="spice"
                      data-sub="powder"
                      >Bột nêm & Gia vị khô</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="spice"
                      data-sub="herbs"
                      >Thảo mộc & Gia vị tươi</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="spice"
                      data-sub="condiment"
                      >Gia vị đặc biệt & Hỗn hợp nấu ăn</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Đồ ngọt -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Đồ Ngọt</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="sweet"
                      data-sub="snack"
                      >Bánh snack</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="sweet"
                      data-sub="chocolate"
                      >Socola & Bánh kẹo socola</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="sweet"
                      data-sub="candy"
                      >Kẹo ngọt</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="sweet"
                      data-sub="dried"
                      >Mứt & Trái cây sấy</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="sweet"
                      data-sub="cereal"
                      >Ngũ cốc & Granola</a
                    >
                  </li>
                </ul>
              </div>

              <!-- Đồ gia dụng -->
              <div class="mega-menu__column">
                <h3 class="mega-menu__title">Đồ Gia Dụng</h3>
                <ul class="mega-menu__list">
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="household"
                      data-sub="paper"
                      >Giấy & Khăn</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="household"
                      data-sub="laundry-cleaning"
                      >Giặt giũ & Vệ sinh</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="household"
                      data-sub="kitchen-storage"
                      >Dụng cụ bếp & Lưu trữ</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="household"
                      data-sub="bags-packaging"
                      >Túi, Bao bì & Rác</a
                    >
                  </li>
                  <li>
                    <a
                      href="${catalogHref}"
                      class="mega-menu__link"
                      data-category="household"
                      data-sub="personal-care"
                      >Chăm sóc cá nhân</a
                    >
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <a href="/html/recipes.html">Công thức</a>
        <a href="/html/recommended.html">Gợi ý cá nhân</a>
        <a href="/html/season.html">Theo mùa</a>
        <a href="/html/aboutus.html">Giới thiệu</a>
      </nav>
      <div class="searchbox" role="search">
        <input id="searchInput" class="searchbox__input" type="search" placeholder="Search" aria-label="Tìm kiếm sản phẩm" autocomplete="on" />
        <button class="searchbox__btn" aria-label="Tìm">${iconSearch}</button>
      </div>
      <div class="actions">
        <a class="btn btn--icon" id="accountBtn" href="../client/register.html">Tài khoản</a>
        <button class="btn btn--pri" id="cartOpenBtn" aria-haspopup="dialog" aria-controls="cartDrawer">Giỏ hàng <span id="cartBadge" class="badge">0</span></button>
      </div>
    </div>`;
}

function mountHeader() {
  const container =
    document.querySelector("#siteHeader") ||
    document.querySelector("header.header");
  if (!container) return;
  container.innerHTML = buildHeaderHTML();
  const isHome = /\/html\/index\.html$/i.test(location.pathname);
  // When header is fixed, add top padding to main equal to header height
  requestAnimationFrame(() => {
    try {
      const h = container.offsetHeight || 60;
      document.documentElement.style.setProperty("--header-h", h + "px");
      document.body.classList.add("has-fixed-header");
    } catch {}
  });
  // Bind mega menu hover/click interactions (links already point đúng bằng catalogHref)
  bindMegaMenu();
  // Update cart badge immediately and on cart changes
  const badge = document.getElementById("cartBadge");
  if (badge) badge.textContent = String(sumCartQty());
  document.addEventListener("cart:changed", () => {
    const b = document.getElementById("cartBadge");
    if (b) b.textContent = String(sumCartQty());
  });

  // Cart button: trên trang khác không có drawer, điều hướng về cart.html
  const cartBtn = document.getElementById("cartOpenBtn");
  if (cartBtn) {
    cartBtn.addEventListener("click", (ev) => {
      ev.preventDefault();
      const url = new URL("/html/cart.html", location.href).toString();
      location.href = url;
    });
  }

  // Account button: nếu đã đăng nhập -> account.html; nếu chưa -> login.html
  const accountBtn = document.getElementById("accountBtn");
  if (accountBtn) {
    // Cập nhật href ban đầu theo trạng thái hiện tại
    (async () => {
      try {
        const u = await apiCurrentUser();
        accountBtn.setAttribute(
          "href",
          u ? "/html/account.html" : "/client/login.html"
        );
      } catch {}
    })();
    accountBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      try {
        const u = await apiCurrentUser();
        const url = new URL(
          u ? "/html/account.html" : "/client/login.html",
          location.href
        );
        location.href = url.toString();
      } catch {
        location.href = new URL("/client/login.html", location.href).toString();
      }
    });
  }

  // Search: từ bất kỳ trang nào, điều hướng về trang sản phẩm
  const searchBtn = document.querySelector(".searchbox__btn");
  const searchInput = document.getElementById("searchInput");
  let _productsCache = null;
  const ensureProducts = async () => {
    if (Array.isArray(_productsCache) && _productsCache.length)
      return _productsCache;
    try {
      _productsCache = await apiListProducts();
    } catch {
      _productsCache = [];
    }
    return _productsCache;
  };
  const goHomeWithQuery = async (ev) => {
    const q = (searchInput?.value || "").trim();
    if (!isHome) {
      ev?.preventDefault?.();
      const url = new URL("/html/index.html#catalog", location.href);
      // Nếu khớp chính xác tên sản phẩm, điều hướng bằng ?product=ID để hiển thị đúng 1 sản phẩm
      if (q) {
        const list = await ensureProducts();
        const found = list.find(
          (p) => (p?.name || "").trim().toLowerCase() === q.toLowerCase()
        );
        if (found && found.id) url.searchParams.set("product", found.id);
        else url.searchParams.set("q", q);
      }
      location.href = url.toString();
    }
    // Nếu đang ở trang chủ, để main.js xử lý lọc mà không reload
  };
  if (searchBtn) searchBtn.addEventListener("click", goHomeWithQuery);
  if (searchInput) {
    searchInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") goHomeWithQuery(ev);
    });
    // Khi chọn từ droplist trên trang khác, điều hướng ngay nếu tên khớp chính xác
    if (!isHome) {
      searchInput.addEventListener("input", async () => {
        const q = (searchInput?.value || "").trim();
        if (!q) return;
        const list = await ensureProducts();
        const found = list.find(
          (p) => (p?.name || "").trim().toLowerCase() === q.toLowerCase()
        );
        if (found && found.id) {
          const url = new URL("/html/index.html#catalog", location.href);
          url.searchParams.set("product", found.id);
          location.href = url.toString();
        }
      });
    }
  }
}

// Mount header càng sớm càng tốt để các module khác có thể bắt DOM refs
(function initHeaderMount() {
  const run = () => {
    try {
      // Đảm bảo nạp giỏ hàng từ localStorage trước khi dựng header và tính badge
      try {
        loadCart();
      } catch {}
      mountHeader();
    } catch {}
  };
  if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);
})();
