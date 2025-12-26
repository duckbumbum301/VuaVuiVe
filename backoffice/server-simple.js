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
import SeasonalRecommender from "./ml/models/SeasonalRecommender.js";

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
const seasonalRecommender = new SeasonalRecommender({
  dbPath: path.join(__dirname, "db.json"),
  minSupport: 0.2,
  minConfidence: 0.5,
});

const PORT = 3000;

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

// ====== SEASONAL ASSOCIATION RULES (S-APRIORI) ======
server.get("/api/recommend/seasonal", async (_req, res) => {
  try {
    const result = await seasonalRecommender.getSeasonalRules();
    res.json(result);
  } catch (err) {
    console.error("Seasonal recommendation error:", err);
    res
      .status(500)
      .json({ error: "Failed to generate seasonal recommendations" });
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
  // Kiểm tra nếu là POST/PUT/PATCH/DELETE /products
  const isProductRoute =
    req.path === "/products" || req.path.startsWith("/products/");
  const isModifyingRequest = ["POST", "PUT", "PATCH", "DELETE"].includes(
    req.method
  );

  if (
    isProductRoute &&
    isModifyingRequest &&
    res.statusCode >= 200 &&
    res.statusCode < 300
  ) {
    try {
      const db = router.db.getState();
      if (db && db.products) {
        const PRODUCTS_FILE = path.join(__dirname, "data", "products.json");
        fs.writeFileSync(
          PRODUCTS_FILE,
          JSON.stringify(db.products, null, 2),
          "utf-8"
        );
        console.log(
          `📝 Synced ${db.products.length} products to data/products.json`
        );
      }
    } catch (error) {
      console.error("❌ Error syncing products:", error.message);
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
