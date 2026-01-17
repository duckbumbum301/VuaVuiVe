// js/api.js — Mock "REST API" chạy trong trình duyệt bằng localStorage
import { PRODUCTS } from "./data.js";

const LS_ORDERS = "vvv_orders";
const LS_USERS = "vvv_users_v1"; // danh sách người dùng (email/phone/pass)
const LS_SESSION = "vvv_session_v1"; // phiên đăng nhập thống nhất (email/phone)
const OLD_LS_USERS = "vvv_users";
const OLD_CLIENT_USERS = "client_users_v1";
const OLD_LS_SESSION = "vvv_session";
const OLD_CLIENT_SESSION = "client_session_v1";

// ---- LocalStorage helpers ----
function lsGet(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback));
  } catch {
    return fallback;
  }
}
function lsSet(key, val) {
  localStorage.setItem(key, JSON.stringify(val));
}

// ---- Migration users (best-effort) ----
(function migrateAuthKeys() {
  try {
    const newUsersRaw = localStorage.getItem(LS_USERS);
    if (!newUsersRaw) {
      const old1 = localStorage.getItem(OLD_LS_USERS);
      const old2 = localStorage.getItem(OLD_CLIENT_USERS);
      if (old1) localStorage.setItem(LS_USERS, old1);
      else if (old2) localStorage.setItem(LS_USERS, old2);
    }
    const newSessRaw = localStorage.getItem(LS_SESSION);
    if (!newSessRaw) {
      const oldS1 = localStorage.getItem(OLD_LS_SESSION);
      const oldS2 = localStorage.getItem(OLD_CLIENT_SESSION);
      if (oldS1) localStorage.setItem(LS_SESSION, oldS1);
      else if (oldS2) localStorage.setItem(LS_SESSION, oldS2);
    }
  } catch {}
})();

// ========= PRODUCTS =========
export async function apiListProducts() {
  try {
    // Load from JSON Server API
    const response = await fetch("http://localhost:3000/products");
    if (!response.ok) {
      console.warn("Failed to fetch from API, falling back to static data");
      return PRODUCTS.map(normalizeProduct);
    }
    const products = await response.json();
    // Filter only active products for frontend and normalize fields
    return products
      .filter((p) => p.status === "active" || !p.status)
      .map(normalizeProduct);
  } catch (error) {
    console.error("Error loading products from API:", error);
    // Fallback to static data if API fails
    return PRODUCTS.map(normalizeProduct);
  }
}

export async function apiGetProductById(id) {
  try {
    const response = await fetch(`http://localhost:3000/products/${id}`);
    if (!response.ok) {
      const found = PRODUCTS.find((p) => p.id === id) || null;
      return found ? normalizeProduct(found) : null;
    }
    const raw = await response.json();
    return normalizeProduct(raw);
  } catch (error) {
    console.error("Error fetching product:", error);
    const found = PRODUCTS.find((p) => p.id === id) || null;
    return found ? normalizeProduct(found) : null;
  }
}

// ---- Helpers: normalize product fields ----
function stripDiacritics(str) {
  try {
    return String(str || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim();
  } catch {
    return String(str || "")
      .toLowerCase()
      .trim();
  }
}

function normalizeCategorySlug(cat) {
  const s = stripDiacritics(cat);
  const known = [
    "veg",
    "fruit",
    "meat",
    "drink",
    "dry",
    "spice",
    "household",
    "sweet",
  ];
  if (known.includes(s)) return s;
  if (s === "rau cu") return "veg";
  if (s === "trai cay") return "fruit";
  if (s === "thit ca") return "meat";
  if (s === "nuoc giai khat" || s === "do uong") return "drink";
  if (s === "do kho") return "dry";
  if (s === "gia vi") return "spice";
  if (s === "do gia dung") return "household";
  if (s === "do ngot") return "sweet";
  return s || "all";
}

function normalizeProduct(p) {
  const catRaw = p.cat ?? p.category ?? "";
  const subRaw = p.sub ?? p.subcategory ?? "";
  const normCat = normalizeCategorySlug(catRaw);
  const normSub = String(subRaw || "").trim() || "all";
  return { ...p, cat: normCat, sub: normSub };
}

// ========= VOUCHERS (mock) =========
export async function apiApplyVoucher(code, { subtotal, shippingFee }) {
  const c = (code || "").trim().toUpperCase();
  if (!c) return { ok: false, message: "Bạn chưa nhập mã." };

  // FREESHIP: miễn phí vận chuyển
  if (c === "FREESHIP")
    return {
      ok: true,
      type: "ship",
      value: shippingFee,
      message: "Đã áp dụng freeship.",
    };

  // GIAM10: giảm 10% tổng hàng
  if (c === "GIAM10")
    return {
      ok: true,
      type: "percent",
      value: 10,
      message: "Giảm 10% đơn hàng.",
    };

  // VUA50: giảm 50% cho đơn đầu tiên của tài khoản (tối đa 500.000đ)
  if (c === "VUA50") {
    // Yêu cầu đăng nhập và kiểm tra lịch sử đơn
    const cur = await apiCurrentUser();
    if (!cur)
      return {
        ok: false,
        message: "Vui lòng đăng nhập để áp dụng mã người mới.",
      };
    const orders = lsGet(LS_ORDERS, []);
    const hasAnyOrder = orders.some((o) => {
      const u = o.user || {};
      const samePhone =
        cur.phone && u.phone && String(u.phone) === String(cur.phone);
      const sameId =
        (o.customer_id && cur.id && String(o.customer_id) === String(cur.id)) ||
        false;
      return samePhone || sameId;
    });
    if (hasAnyOrder)
      return {
        ok: false,
        message: "Mã VUA50 chỉ áp dụng cho đơn đầu tiên của tài khoản.",
      };

    const CAP = 500000; // tối đa 500.000đ
    return {
      ok: true,
      type: "percent",
      value: 50,
      cap: CAP,
      message: "Giảm 50% cho đơn đầu tiên (tối đa 500.000đ).",
    };
  }

  return { ok: false, message: "Mã không hợp lệ." };
}

// ========= DELIVERY SLOTS (mock) =========
export async function apiListDeliverySlots() {
  const slots = [];
  const today = new Date();
  for (let d = 0; d < 2; d++) {
    const date = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() + d
    );
    ["09:00-11:00", "13:00-15:00", "18:00-20:00"].forEach((win) => {
      slots.push({
        id: `${date.toISOString().slice(0, 10)}_${win}`,
        date: date.toISOString().slice(0, 10),
        window: win,
        capacity: 50,
        used: 0,
      });
    });
  }
  return slots;
}

// ========= SHIPPING (mock) =========
export function calcShippingFee(addressText, subtotal) {
  const txt = (addressText || "").toLowerCase();
  if (!txt) return 20000;
  let fee = /q\.\s*\d+|quận|tp\./.test(txt)
    ? 15000
    : /h\.\s*|huyện/.test(txt)
      ? 25000
      : 20000;
  if (subtotal >= 300000) fee = 0;
  return fee;
}

// Helper: Sinh mã đơn hàng chuẩn thời gian + random (ORD-YYYYMMDD-HHmmss-RAND)
function genOrderId() {
  const now = new Date();
  const ymd = now.toISOString().slice(0, 10).replace(/-/g, "");
  const hms = `${String(now.getHours()).padStart(2, "0")}${String(
    now.getMinutes()
  ).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;
  const rand = Math.random().toString(36).slice(2, 8).toUpperCase();
  return `ORD-${ymd}-${hms}-${rand}`;
}

// ========= ORDERS =========
const API_BASE = "http://localhost:3000";

export async function apiCreateOrder(orderPayload) {
  // Validate: không cho tạo đơn nếu không có item hoặc tổng tiền = 0
  const items = orderPayload?.items || {};
  const hasItems = Object.values(items).some((q) => Number(q) > 0);
  const subtotalNum = Number(orderPayload?.subtotal || 0);
  const totalNum = Number(orderPayload?.total || 0);
  if (!hasItems || subtotalNum <= 0 || totalNum <= 0) {
    throw new Error("EMPTY_ORDER");
  }

  const id = genOrderId();
  const createdAt = new Date().toISOString();
  const tracking = [
    { code: "placed", label: "Đã đặt hàng", at: createdAt },
    { code: "preparing", label: "Đang chuẩn bị", at: null },
    { code: "ready", label: "Sẵn sàng giao", at: null },
    { code: "pickup", label: "Shipper đã nhận", at: null },
    { code: "delivering", label: "Đang giao", at: null },
    { code: "delivered", label: "Giao thành công", at: null },
  ];
  const newOrder = {
    id,
    ...orderPayload,
    created_at: createdAt,
    tracking,
    // Respect caller-provided statuses (e.g. COD = unpaid, VNPAY = pending)
    payment_status: orderPayload?.payment_status ?? "pending",
    delivery_status: orderPayload?.delivery_status ?? "placed",
  };

  try {
    // POST directly to json-server /orders
    // Middleware sẽ tự động trừ stock
    const response = await fetch(`${API_BASE}/orders`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(newOrder),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || errorData.message || "API failed");
    }

    const savedOrder = await response.json();
    console.log("✅ Order created successfully:", savedOrder.id);

    // Log audit
    const user = await apiCurrentUser();
    if (user) {
      await apiCreateAuditLog("order.create", user.email || user.phone, {
        orderId: savedOrder.id,
        total: savedOrder.total,
      });
    }

    // Stock is automatically deducted by backend middleware
    return savedOrder;
  } catch (error) {
    console.error("Failed to create order via API:", error.message);

    // Fallback to localStorage only if network error
    if (error.message.includes("fetch")) {
      console.warn("Using localStorage fallback due to network error");
      const orders = lsGet(LS_ORDERS, []);
      orders.push(newOrder);
      lsSet(LS_ORDERS, orders);
      return newOrder;
    }

    // Re-throw validation errors (like out of stock)
    throw error;
  }
}

export async function apiListOrders(filters = {}) {
  try {
    let url = `${API_BASE}/orders?_sort=created_at&_order=desc`;
    if (filters.status) url += `&delivery_status=${filters.status}`;
    if (filters.customer_id) url += `&user.id=${filters.customer_id}`;

    const response = await fetch(url);
    if (!response.ok) throw new Error("API failed");
    return await response.json();
  } catch (error) {
    console.error(
      "Failed to fetch orders from API, using localStorage fallback:",
      error
    );
    return lsGet(LS_ORDERS, []);
  }
}

export async function apiGetOrderById(orderId) {
  try {
    const response = await fetch(`${API_BASE}/orders/${orderId}`);
    if (!response.ok) throw new Error("Order not found");
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch order from API:", error);
    const orders = lsGet(LS_ORDERS, []);
    return orders.find((o) => o.id === orderId) || null;
  }
}

export async function apiUpdateOrderStatus(
  orderId,
  newStatus,
  updatedBy = "Admin"
) {
  try {
    const order = await apiGetOrderById(orderId);
    if (!order) throw new Error("Order not found");

    const now = new Date().toISOString();
    const tracking = Array.isArray(order.tracking) ? [...order.tracking] : [];

    // Update tracking based on new status
    const trackingMap = {
      preparing: { code: "preparing", label: "Đang chuẩn bị", at: now },
      ready: { code: "ready", label: "Sẵn sàng giao", at: now },
      pickup: { code: "pickup", label: "Shipper đã nhận", at: now },
      delivering: { code: "delivering", label: "Đang giao", at: now },
      delivered: { code: "delivered", label: "Giao thành công", at: now },
      cancelled: { code: "cancelled", label: "Đã hủy đơn", at: now },
      returned: { code: "returned", label: "Đã trả hàng", at: now },
    };

    const trackingEntry = trackingMap[newStatus];
    if (trackingEntry) {
      const existingIdx = tracking.findIndex(
        (t) => t.code === trackingEntry.code
      );
      if (existingIdx >= 0) {
        tracking[existingIdx].at = now;
      } else {
        tracking.push(trackingEntry);
      }
    }

    const updates = {
      delivery_status: newStatus,
      tracking,
      updated_at: now,
      updated_by: updatedBy,
    };

    const response = await fetch(`${API_BASE}/orders/${orderId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error("Update failed");
    const updatedOrder = await response.json();

    // Create audit log
    await apiCreateAuditLog("update_order_status", updatedBy, {
      orderId,
      oldStatus: order.delivery_status,
      newStatus,
    });

    return updatedOrder;
  } catch (error) {
    console.error("Failed to update order via API:", error);
    // Fallback to localStorage
    const orders = lsGet(LS_ORDERS, []);
    const idx = orders.findIndex((o) => o.id === orderId);
    if (idx === -1) return null;
    const now = new Date().toISOString();
    const o = orders[idx];
    o.delivery_status = newStatus;
    o.tracking = Array.isArray(o.tracking) ? o.tracking.slice() : [];
    const has = o.tracking.some((s) => s.code === newStatus);
    if (!has) o.tracking.push({ code: newStatus, label: newStatus, at: now });
    lsSet(LS_ORDERS, orders);
    return o;
  }
}

export async function apiCancelOrder(orderId, cancelledBy = "Customer") {
  return await apiUpdateOrderStatus(orderId, "cancelled", cancelledBy);
}

export async function apiReturnOrder(orderId, returnedBy = "Customer") {
  return await apiUpdateOrderStatus(orderId, "returned", returnedBy);
}

// Đánh dấu đơn đã thanh toán (giả lập online)
export async function apiMarkOrderPaid(orderId) {
  try {
    // Gọi endpoint mới để xử lý thanh toán và trừ stock
    const response = await fetch(`${API_BASE}/api/orders/${orderId}/paid`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
    });
    if (!response.ok) throw new Error("Update failed");

    // Create audit log
    await apiCreateAuditLog("mark_paid", "System", { orderId });

    return await response.json();
  } catch (error) {
    console.error("Failed to mark order paid via API:", error);
    // Fallback to localStorage
    const orders = lsGet(LS_ORDERS, []);
    const idx = orders.findIndex((o) => o.id === orderId);
    if (idx === -1) return null;
    const o = orders[idx];
    o.payment_status = "paid";
    o.paid_at = new Date().toISOString();
    lsSet(LS_ORDERS, orders);
    return o;
  }
}

// Đánh dấu thanh toán thất bại
export async function apiMarkOrderPaymentFailed(
  orderId,
  reason = "Payment failed"
) {
  try {
    const response = await fetch(
      `${API_BASE}/api/orders/${orderId}/payment-failed`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      }
    );
    if (!response.ok) throw new Error("Update failed");

    // Create audit log
    await apiCreateAuditLog("mark_payment_failed", "System", {
      orderId,
      reason,
    });

    return await response.json();
  } catch (error) {
    console.error("Failed to mark order payment failed via API:", error);
    // Fallback to localStorage
    const orders = lsGet(LS_ORDERS, []);
    const idx = orders.findIndex((o) => o.id === orderId);
    if (idx === -1) return null;
    const o = orders[idx];
    o.payment_status = "failed";
    o.status = "cancelled";
    o.delivery_status = "cancelled";
    o.payment_failed_reason = reason;
    lsSet(LS_ORDERS, orders);
    return o;
  }
}

// ========= AUTH / USERS =========
function getUsers() {
  try {
    const u = JSON.parse(localStorage.getItem(LS_USERS) || "[]");
    if (Array.isArray(u)) return u;
    return [];
  } catch {
    return [];
  }
}
function setUsers(list) {
  lsSet(LS_USERS, list);
}

// ====== Seed tài khoản test cố định (phục vụ demo) ======
(function seedFixedTestUser() {
  try {
    const users = getUsers();
    const idx = users.findIndex((u) => (u.phone || "") === "0906760495");
    if (idx === -1) {
      const testUser = {
        id: "seed-0906760495",
        name: "Nguyễn Chí Đức",
        email: "ilovetranduythanh@gmail.com",
        phone: "0906760495",
        password: "10diem10diem",
        address: "669 Đỗ Mười, khu phố 13, phường Linh Xuân, TP.HCM",
      };
      users.push(testUser);
      setUsers(users);
    } else {
      // Cập nhật để đảm bảo tài khoản mặc định luôn đúng thông tin yêu cầu
      users[idx] = {
        ...users[idx],
        name: "Nguyễn Chí Đức",
        email: "ilovetranduythanh@gmail.com",
        phone: "0906760495",
        password: "10diem10diem",
        address: "669 Đỗ Mười, khu phố 13, phường Linh Xuân, TP.HCM",
      };
      setUsers(users);
    }
    // Không tự đăng nhập; chỉ tạo/đồng bộ tài khoản
  } catch {}
})();

export async function apiRegisterUser({
  name,
  email,
  phone,
  password,
  address,
}) {
  const e = (email || "").trim().toLowerCase();
  const p = (phone || "").replace(/\D/g, "");
  const users = getUsers();
  if (!name || (!e && !p) || !password)
    return {
      ok: false,
      message: "Vui lòng nhập đủ họ tên, SĐT/email và mật khẩu.",
    };
  if (e && users.some((u) => (u.email || "").toLowerCase() === e))
    return { ok: false, message: "Email đã tồn tại." };
  if (p && users.some((u) => (u.phone || "") === p))
    return { ok: false, message: "SĐT đã tồn tại." };
  const id = Date.now().toString();
  const user = {
    id,
    name: name.trim(),
    email: e || "",
    phone: p || "",
    password: String(password),
    address: (address || "").trim(),
  };
  users.push(user);
  setUsers(users);

  // KHÔNG tự động tạo session sau khi đăng ký - người dùng phải đăng nhập lại
  // lsSet(LS_SESSION, {
  //   id: user.id,
  //   email: user.email,
  //   phone: user.phone,
  //   name: user.name,
  // });

  // Log audit
  await apiCreateAuditLog("user.register", user.email || user.phone, {
    userId: user.id,
  });

  return {
    ok: true,
    user: {
      id: user.id,
      name: user.name,
      email: user.email,
      phone: user.phone,
      address: user.address,
    },
  };
}

export async function apiLoginUser({ email, phone, password }) {
  const users = getUsers();
  const e = (email || "").trim().toLowerCase();
  const p = (phone || "").replace(/\D/g, "");
  const pw = String(password || "");
  const candidate = users.find(
    (x) =>
      (e && (x.email || "").toLowerCase() === e) || (p && (x.phone || "") === p)
  );
  if (!candidate) {
    return {
      ok: false,
      reason: "user_not_found",
      message: "Không tìm thấy tài khoản. Vui lòng đăng ký.",
    };
  }
  if (candidate.password !== pw) {
    return {
      ok: false,
      reason: "wrong_password",
      message: "Mật khẩu không đúng.",
    };
  }
  lsSet(LS_SESSION, {
    id: candidate.id,
    email: candidate.email,
    phone: candidate.phone,
    name: candidate.name,
  });

  // Log audit
  await apiCreateAuditLog("user.login", candidate.email || candidate.phone, {
    userId: candidate.id,
  });

  return {
    ok: true,
    user: {
      id: candidate.id,
      name: candidate.name,
      email: candidate.email,
      phone: candidate.phone,
      address: candidate.address,
    },
  };
}

/**
 * Đọc phiên hiện tại theo thứ tự ưu tiên:
 * 1) Phiên thống nhất LS_SESSION (vvv_session_v1)
 * 2) Fallback: client_session_v1
 * 3) Fallback: vvv_session (cũ)
 */
export async function apiCurrentUser() {
  try {
    const s = JSON.parse(localStorage.getItem(LS_SESSION) || "null");
    if (s && (s.id || s.phone || s.email)) {
      const users = getUsers();
      let u = null;
      if (s.id) u = users.find((x) => x.id === s.id);
      if (!u && s.email)
        u = users.find(
          (x) => (x.email || "").toLowerCase() === String(s.email).toLowerCase()
        );
      if (!u && s.phone)
        u = users.find((x) => (x.phone || "") === String(s.phone));
      if (u)
        return {
          id: u.id,
          name: u.name,
          email: u.email,
          phone: u.phone,
          address: u.address,
        };
      // nếu không có trong users, trả về thông tin tối thiểu từ session
      return {
        id: s.id || s.phone || s.email,
        name: s.name || "",
        email: s.email || "",
        phone: s.phone || "",
        address: s.address || "",
      };
    }
  } catch {}
  // Fallback client
  try {
    const c = JSON.parse(localStorage.getItem(OLD_CLIENT_SESSION) || "null");
    if (c && (c.phone || c.email)) {
      return {
        id: c.phone || c.email,
        name: c.name || "",
        email: c.email || "",
        phone: c.phone || "",
        address: c.address || "",
      };
    }
  } catch {}
  // Fallback nội bộ cũ
  const sOld = lsGet(OLD_LS_SESSION, null);
  if (!sOld) return null;
  const users = getUsers();
  const uOld = users.find((x) => x.id === sOld.id);
  return uOld
    ? {
        id: uOld.id,
        name: uOld.name,
        email: uOld.email,
        phone: uOld.phone,
        address: uOld.address,
      }
    : null;
}

export async function apiLogoutUser() {
  localStorage.removeItem(LS_SESSION);
  localStorage.removeItem(OLD_CLIENT_SESSION);
  localStorage.removeItem(OLD_LS_SESSION);
  // Đồng bộ luồng Account: clear cờ đã xem profile khi logout qua API
  try {
    localStorage.removeItem("vvv_has_seen_profile");
  } catch {}
  return { ok: true };
}

export async function apiUpdateProfile({ name, address, phone, email }) {
  // Ưu tiên session mới
  const sRaw = localStorage.getItem(LS_SESSION);
  let s = null;
  try {
    s = JSON.parse(sRaw || "null");
  } catch {}
  if (!s) return { ok: false, message: "Chưa đăng nhập." };
  const users = getUsers();
  let idx = -1;
  if (s.id) idx = users.findIndex((x) => x.id === s.id);
  if (idx === -1 && s.email)
    idx = users.findIndex(
      (x) => (x.email || "").toLowerCase() === String(s.email).toLowerCase()
    );
  if (idx === -1 && s.phone)
    idx = users.findIndex((x) => (x.phone || "") === String(s.phone));
  if (idx === -1) return { ok: false, message: "Không tìm thấy người dùng." };

  // Chuẩn hóa & kiểm tra trùng SĐT nếu có yêu cầu cập nhật
  let newPhone = (phone ?? users[idx].phone) || "";
  newPhone = String(newPhone).replace(/\D/g, "");
  if (newPhone && newPhone !== (users[idx].phone || "")) {
    const dupPhone = users.some(
      (u, i) => i !== idx && (u.phone || "") === newPhone
    );
    if (dupPhone) return { ok: false, message: "SĐT đã tồn tại." };
  }

  // Chuẩn hóa & kiểm tra trùng Email nếu có yêu cầu cập nhật
  let newEmail = (email ?? users[idx].email) || "";
  newEmail = String(newEmail).trim().toLowerCase();
  if (newEmail && newEmail !== (users[idx].email || "").toLowerCase()) {
    const dupEmail = users.some(
      (u, i) => i !== idx && String(u.email || "").toLowerCase() === newEmail
    );
    if (dupEmail) return { ok: false, message: "Email đã tồn tại." };
  }

  users[idx] = {
    ...users[idx],
    name: name || users[idx].name,
    address: address || users[idx].address,
    phone: newPhone,
    email: newEmail,
  };
  setUsers(users);
  lsSet(LS_SESSION, {
    id: users[idx].id,
    email: users[idx].email,
    phone: users[idx].phone,
    name: users[idx].name,
  });

  // Log audit
  await apiCreateAuditLog(
    "profile.update",
    users[idx].email || users[idx].phone,
    { userId: users[idx].id, changes: { name, address, phone, email } }
  );

  return {
    ok: true,
    user: {
      id: users[idx].id,
      name: users[idx].name,
      email: users[idx].email,
      phone: users[idx].phone,
      address: users[idx].address,
    },
  };
}

export async function apiChangePassword({ oldPassword, newPassword }) {
  // Đảm bảo đang đăng nhập
  let s = null;
  try {
    s = JSON.parse(localStorage.getItem(LS_SESSION) || "null");
  } catch {}
  if (!s) return { ok: false, message: "Chưa đăng nhập." };

  const users = getUsers();
  let idx = -1;
  if (s.id) idx = users.findIndex((x) => x.id === s.id);
  if (idx === -1 && s.email)
    idx = users.findIndex(
      (x) => (x.email || "").toLowerCase() === String(s.email).toLowerCase()
    );
  if (idx === -1 && s.phone)
    idx = users.findIndex((x) => (x.phone || "") === String(s.phone));
  if (idx === -1) return { ok: false, message: "Không tìm thấy người dùng." };

  const curPw = String(oldPassword || "");
  const newPw = String(newPassword || "");
  if (!curPw) return { ok: false, message: "Vui lòng nhập mật khẩu hiện tại." };
  if (users[idx].password !== curPw)
    return {
      ok: false,
      reason: "wrong_old_password",
      message: "Mật khẩu hiện tại không đúng.",
    };
  if (!newPw || newPw.length < 6)
    return {
      ok: false,
      reason: "weak_password",
      message: "Mật khẩu mới phải ≥ 6 ký tự.",
    };
  if (newPw === curPw)
    return {
      ok: false,
      message: "Mật khẩu mới không được trùng mật khẩu hiện tại.",
    };

  users[idx] = { ...users[idx], password: newPw };
  setUsers(users);

  // Log audit
  await apiCreateAuditLog(
    "password.change",
    users[idx].email || users[idx].phone,
    { userId: users[idx].id }
  );

  // Không cần cập nhật session vì phiên không lưu password
  return { ok: true };
}

// ========= AUDIT LOGS =========
const LS_AUDIT = "vvv_audit_logs";

export async function apiCreateAuditLog(action, who = "System", metadata = {}) {
  const logEntry = {
    id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
    action,
    who,
    metadata,
    timestamp: new Date().toISOString(),
  };

  try {
    const response = await fetch(`${API_BASE}/auditLogs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(logEntry),
    });
    if (!response.ok) throw new Error("Failed to create audit log");
    return await response.json();
  } catch (error) {
    console.error(
      "Failed to create audit log via API, using localStorage:",
      error
    );
    const logs = lsGet(LS_AUDIT, []);
    logs.push(logEntry);
    lsSet(LS_AUDIT, logs);
    return logEntry;
  }
}

export async function apiListAuditLogs(limit = 100) {
  try {
    const response = await fetch(
      `${API_BASE}/auditLogs?_sort=timestamp&_order=desc&_limit=${limit}`
    );
    if (!response.ok) throw new Error("Failed to fetch audit logs");
    return await response.json();
  } catch (error) {
    console.error(
      "Failed to fetch audit logs from API, using localStorage:",
      error
    );
    const logs = lsGet(LS_AUDIT, []);
    return logs.slice().reverse().slice(0, limit);
  }
}
