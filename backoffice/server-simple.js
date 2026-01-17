// server-simple.js - Json-server với middleware trừ stock
import jsonServer from "json-server";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  stockDeductionMiddleware,
  stockRestoreMiddleware,
  productSyncMiddleware,
} from "./server-middleware.js";
import TimeAwareRecommender from "./ml/models/TimeAwareRecommender.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const server = jsonServer.create();
const router = jsonServer.router(path.join(__dirname, "db.json"));
const middlewares = jsonServer.defaults();
const recommender = new TimeAwareRecommender({
  dbPath: path.join(__dirname, "db.json"),
  lambda: 0.05,
  lookbackDays: 90,
});

const PORT = 3000;

// ====== SYNC json-server db.json -> backoffice/data/*.json ======
// NOTE: data/orders.json is also consumed by the ML adapter; normalize items to array format.
const DATA_DIR = path.join(__dirname, "data");
const DATA_FILES = {
  products: path.join(DATA_DIR, "products.json"),
  orders: path.join(DATA_DIR, "orders.json"),
  users: path.join(DATA_DIR, "users.json"),
  auditLogs: path.join(DATA_DIR, "auditLogs.json"),
};

function safeWriteJSON(filePath, data) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
    return true;
  } catch (error) {
    console.error(
      `❌ Error syncing ${path.basename(filePath)}:`,
      error.message
    );
    return false;
  }
}

function normalizeOrderItems(items, productsById) {
  if (Array.isArray(items)) {
    return items
      .map((item) => {
        const productId = String(item?.productId ?? item?.id ?? "");
        const quantity = Number.parseInt(item?.quantity ?? item?.qty ?? 0, 10);
        const product = productsById.get(productId);
        const price = Number(item?.price ?? product?.price ?? 0);
        const productName =
          item?.productName ?? item?.name ?? product?.name ?? "";
        const subtotal = Number(item?.subtotal ?? price * quantity);
        return { productId, productName, quantity, price, subtotal };
      })
      .filter((i) => i.productId && Number(i.quantity) > 0);
  }

  if (items && typeof items === "object") {
    return Object.entries(items)
      .map(([productId, quantityRaw]) => {
        const quantity = Number.parseInt(quantityRaw, 10);
        const product = productsById.get(String(productId));
        const price = Number(product?.price ?? 0);
        const productName = product?.name ?? "";
        const subtotal = price * quantity;
        return {
          productId: String(productId),
          productName,
          quantity,
          price,
          subtotal,
        };
      })
      .filter((i) => i.productId && Number(i.quantity) > 0);
  }

  return [];
}

function normalizeOrdersForData(dbState) {
  const products = Array.isArray(dbState?.products) ? dbState.products : [];
  const productsById = new Map(products.map((p) => [String(p.id), p]));
  const orders = Array.isArray(dbState?.orders) ? dbState.orders : [];

  return orders.map((order) => {
    const normalizedItems = normalizeOrderItems(order?.items, productsById);
    const computedTotal = normalizedItems.reduce(
      (sum, item) => sum + Number(item.subtotal || 0),
      0
    );

    const createdAt =
      order?.createdAt ?? order?.created_at ?? order?.created_at ?? null;
    const updatedAt =
      order?.updatedAt ?? order?.updated_at ?? order?.updatedAt ?? null;
    const status =
      order?.status ?? order?.delivery_status ?? order?.deliveryStatus ?? null;

    const customerName =
      order?.customerName ?? order?.user?.name ?? order?.customer?.name ?? null;
    const email = order?.email ?? order?.user?.email ?? null;
    const phone = order?.phone ?? order?.user?.phone ?? null;
    const address = order?.address ?? order?.shippingAddress ?? null;

    const totalAmountRaw =
      order?.totalAmount ?? order?.total ?? order?.total_amount ?? undefined;
    const totalAmount = Number(totalAmountRaw ?? computedTotal);

    return {
      ...order,
      customerName,
      email,
      phone,
      address,
      items: normalizedItems,
      totalAmount,
      status,
      createdAt,
      updatedAt,
    };
  });
}

function syncDbToDataFiles(dbState) {
  if (!dbState || typeof dbState !== "object") return;

  const products = Array.isArray(dbState.products) ? dbState.products : [];
  const orders = normalizeOrdersForData(dbState);
  const users = Array.isArray(dbState.users) ? dbState.users : [];
  const auditLogs = Array.isArray(dbState.auditLogs) ? dbState.auditLogs : [];

  safeWriteJSON(DATA_FILES.products, products);
  safeWriteJSON(DATA_FILES.orders, orders);
  safeWriteJSON(DATA_FILES.users, users);
  safeWriteJSON(DATA_FILES.auditLogs, auditLogs);
}

// Expose database to middleware
server.use((req, res, next) => {
  req.app.db = router.db.getState();
  next();
});

// Middlewares
server.use(middlewares);
server.use(jsonServer.bodyParser);

// Logging middleware
server.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// ====== CUSTOM API ENDPOINTS (PHẢI ĐẶT TRƯỚC ROUTER) ======

// ✅ API: Đánh dấu đơn hàng đã thanh toán (VNPay success)
server.patch("/api/orders/:id/paid", (req, res) => {
  const { id } = req.params;
  const db = router.db.getState();
  const order = db.orders.find((o) => o.id === id);

  if (!order) {
    return res.status(404).json({ error: "Order not found" });
  }

  console.log(`💳 Processing VNPay payment for order ${id}...`);

  // ✅ Cập nhật trạng thái
  order.payment_status = "paid";
  order.paid_at = new Date().toISOString();
  order.updatedAt = new Date().toISOString();

  // ✅ TRỪ STOCK (VNPay success)
  let itemsToProcess = [];

  // Support both array and object formats
  if (Array.isArray(order.items)) {
    itemsToProcess = order.items;
  } else if (typeof order.items === "object") {
    // Convert object {"231": 1} to array [{productId: "231", quantity: 1}]
    itemsToProcess = Object.entries(order.items).map(
      ([productId, quantity]) => ({
        productId,
        quantity: parseInt(quantity, 10),
      })
    );
  }

  itemsToProcess.forEach((item) => {
    const product = db.products.find(
      (p) => p.id === (item.productId || item.id)
    );
    if (product) {
      const oldStock = product.stock || 0;
      const qty = item.quantity || 1;
      product.stock = Math.max(0, oldStock - qty);
      console.log(
        `📦 Stock deducted: ${product.name} (${oldStock} → ${product.stock}, -${qty})`
      );
    }
  });

  router.db.write();
  syncDbToDataFiles(router.db.getState());
  console.log(`✅ Order ${id} marked as PAID + stock deducted`);

  res.json({ success: true, order });
});

// ❌ API: Đánh dấu thanh toán thất bại (VNPay failed)
server.patch("/api/orders/:id/payment-failed", (req, res) => {
  const { id } = req.params;
  const { reason } = req.body;
  const db = router.db.getState();
  const order = db.orders.find((o) => o.id === id);

  if (!order) {
    return res.status(404).json({ error: "Order not found" });
  }

  console.log(`❌ Processing VNPay payment failed for order ${id}...`);

  // ❌ Cập nhật trạng thái
  order.payment_status = "failed";
  order.status = "cancelled";
  order.delivery_status = "cancelled";
  order.payment_failed_reason = reason || "Khách hàng hủy giao dịch";
  order.payment_failed_at = new Date().toISOString();
  order.updatedAt = new Date().toISOString();

  // ❌ KHÔNG TRỪ STOCK (VNPay failed)

  router.db.write();
  syncDbToDataFiles(router.db.getState());
  console.log(`❌ Order ${id} marked as PAYMENT FAILED (stock NOT deducted)`);

  res.json({ success: true, order });
});

// ====== RECOMMENDATION API (TIME-AWARE) ======
server.get("/api/recommend/:userId", async (req, res) => {
  const { userId } = req.params;

  try {
    const recommendations = await recommender.getRecommendations(userId);
    res.json({ userId, recommendations });
  } catch (err) {
    console.error("Recommendation error:", err);
    res.status(500).json({ error: "Failed to generate recommendations" });
  }
});

// 🔄 API: Admin cập nhật trạng thái đơn hàng (COD)
server.patch("/orders/:id", (req, res, next) => {
  const { id } = req.params;
  const { delivery_status } = req.body;
  const db = router.db.getState();
  const order = db.orders.find((o) => o.id === id);

  if (!order) {
    return next(); // Let json-server handle 404
  }

  console.log(`🔄 Admin updating order ${id}: ${delivery_status}`);

  // ✅ Nếu COD và admin đánh dấu "Hoàn tất" (delivered) → payment_status = "paid"
  if (order.paymentMethod === "COD" && delivery_status === "delivered") {
    req.body.payment_status = "paid";
    req.body.paid_at = new Date().toISOString();
    console.log(`✅ COD Order ${id} delivered → marked as PAID`);
  }

  // ❌ Nếu admin hủy đơn → payment_status = "cancelled"
  if (delivery_status === "cancelled") {
    req.body.payment_status = "cancelled";
    req.body.status = "cancelled";
    console.log(`❌ Order ${id} cancelled by admin`);
  }

  // Gọi json-server xử lý tiếp
  next();
});

// ====== STOCK MIDDLEWARE (SAU CUSTOM API) ======
server.use(stockDeductionMiddleware);
server.use(stockRestoreMiddleware);
server.use(productSyncMiddleware);

// Custom render để sync sau khi json-server xử lý
router.render = (req, res) => {
  const isModifyingRequest = ["POST", "PUT", "PATCH", "DELETE"].includes(
    req.method
  );
  if (isModifyingRequest && res.statusCode >= 200 && res.statusCode < 300) {
    const root = (req.path.split("/")[1] || "").trim();
    const shouldSync = ["products", "orders", "users", "auditLogs"].includes(
      root
    );

    // Also sync after order mutations because stock changes happen via middleware.
    if (shouldSync) {
      try {
        syncDbToDataFiles(router.db.getState());
      } catch (error) {
        console.error("❌ Error syncing data files:", error.message);
      }
    }
  }

  res.jsonp(res.locals.data);
};

// ====== RSS CORS Proxy (GET /proxy/rss?url=...) ======
server.get("/proxy/rss", async (req, res) => {
  const url = req.query.url;
  if (!url) {
    return res.status(400).json({ error: "Missing url query" });
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    const upstream = await fetch(url, {
      signal: controller.signal,
      headers: {
        // Một số nguồn RSS yêu cầu UA hợp lệ
        "user-agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        accept: "application/rss+xml,text/xml,application/xml;q=0.9,*/*;q=0.8",
      },
    });
    clearTimeout(timeout);

    if (!upstream.ok) {
      return res
        .status(upstream.status)
        .json({ error: `Upstream ${upstream.status} ${upstream.statusText}` });
    }

    const contentType =
      upstream.headers.get("content-type") ||
      "application/rss+xml; charset=utf-8";
    const buffer = Buffer.from(await upstream.arrayBuffer());

    // CORS + cache
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Cache-Control", "public, max-age=300"); // cache 5 phút
    res.setHeader("Content-Type", contentType);
    return res.status(200).send(buffer);
  } catch (err) {
    return res
      .status(502)
      .json({ error: "Proxy fetch failed", detail: err.message });
  }
});

server.use(router);

server.listen(PORT, () => {
  console.log(`\n JSON Server: http://localhost:${PORT}`);
  console.log(` Products: http://localhost:${PORT}/products`);
  console.log(` Orders: http://localhost:${PORT}/orders`);
  console.log(` Users: http://localhost:${PORT}/users\n`);
  console.log(
    ` RSS Proxy: http://localhost:${PORT}/proxy/rss?url=<encoded_url>`
  );
});
