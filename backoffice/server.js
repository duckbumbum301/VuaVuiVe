// backoffice/server.js - REST API Server cho Data Manager
// Sử dụng Express để tạo API endpoints tương tác với data folder

import express from "express";
import cors from "cors";
import dataManager from "./dataManager.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// For ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import recommendation service using dynamic import
let getRecommendationService;
(async () => {
  const module = await import(
    "./ml/services/recommendation/recommendation_service.js"
  );
  getRecommendationService = module.getRecommendationService;
})();

const { productsAPI, ordersAPI, usersAPI, auditLogsAPI, statsAPI } =
  dataManager;

// Helper function to load ML orders
import { promises as fs } from "fs";
async function loadMLOrders() {
  try {
    const mlOrdersPath = join(__dirname, "data", "orders-ml.json");
    const data = await fs.readFile(mlOrdersPath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    console.warn("⚠️ orders-ml.json not found, using regular orders");
    return await ordersAPI.getAll();
  }
}

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Request logging
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// ============ PRODUCTS ENDPOINTS ============

// GET /api/products - Lấy tất cả sản phẩm
app.get("/api/products", async (req, res) => {
  try {
    const { category, status, search } = req.query;
    const products = await productsAPI.getAll({ category, status, search });
    res.json(products);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/products/:id - Lấy sản phẩm theo ID
app.get("/api/products/:id", async (req, res) => {
  try {
    const product = await productsAPI.getById(req.params.id);
    if (!product) {
      return res.status(404).json({ error: "Product not found" });
    }
    res.json(product);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/products - Tạo sản phẩm mới
app.post("/api/products", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const product = await productsAPI.create(req.body, user);
    res.status(201).json(product);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PUT /api/products/:id - Cập nhật sản phẩm
app.put("/api/products/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const product = await productsAPI.update(req.params.id, req.body, user);
    res.json(product);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/products/:id - Cập nhật một phần sản phẩm
app.patch("/api/products/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const product = await productsAPI.update(req.params.id, req.body, user);
    res.json(product);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// DELETE /api/products/:id - Xóa sản phẩm
app.delete("/api/products/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    await productsAPI.delete(req.params.id, user);
    res.json({ success: true, message: "Product deleted" });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/products/:id/stock - Cập nhật stock
app.patch("/api/products/:id/stock", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const { quantity } = req.body;
    const product = await productsAPI.updateStock(
      req.params.id,
      Number(quantity),
      user
    );
    res.json(product);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// ============ ORDERS ENDPOINTS ============

// GET /api/orders - Lấy tất cả đơn hàng
app.get("/api/orders", async (req, res) => {
  try {
    const { status, dateFrom, dateTo, customerId } = req.query;
    const orders = await ordersAPI.getAll({
      status,
      dateFrom,
      dateTo,
      customerId,
    });
    res.json(orders);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/orders/:id - Lấy đơn hàng theo ID
app.get("/api/orders/:id", async (req, res) => {
  try {
    const order = await ordersAPI.getById(req.params.id);
    if (!order) {
      return res.status(404).json({ error: "Order not found" });
    }
    res.json(order);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/orders - Tạo đơn hàng mới
app.post("/api/orders", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "Customer";
    const order = await ordersAPI.create(req.body, user);
    res.status(201).json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/orders/:id/status - Cập nhật trạng thái
app.patch("/api/orders/:id/status", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const { status } = req.body;
    const order = await ordersAPI.updateStatus(req.params.id, status, user);
    res.json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PUT /api/orders/:id - Cập nhật đơn hàng
app.put("/api/orders/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const order = await ordersAPI.update(req.params.id, req.body, user);
    res.json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/orders/:id - Cập nhật một phần đơn hàng
app.patch("/api/orders/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    const order = await ordersAPI.update(req.params.id, req.body, user);
    res.json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// DELETE /api/orders/:id - Xóa đơn hàng
app.delete("/api/orders/:id", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "System";
    await ordersAPI.delete(req.params.id, user);
    res.json({ success: true, message: "Order deleted" });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/orders/:id/paid - Đánh dấu đã thanh toán và trừ stock
app.patch("/api/orders/:id/paid", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "PaymentSystem";
    const order = await ordersAPI.markAsPaid(req.params.id, user);
    res.json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/orders/:id/payment-failed - Đánh dấu thanh toán thất bại
app.patch("/api/orders/:id/payment-failed", async (req, res) => {
  try {
    const user = req.headers["x-user"] || "PaymentSystem";
    const { reason } = req.body;
    const order = await ordersAPI.markAsPaymentFailed(
      req.params.id,
      reason || "Payment failed",
      user
    );
    res.json(order);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// ============ USERS ENDPOINTS ============

// GET /api/users - Lấy tất cả users
app.get("/api/users", async (req, res) => {
  try {
    const users = await usersAPI.getAll();
    // Remove sensitive data if needed
    const safeUsers = users.map((u) => ({
      id: u.id,
      email: u.email,
      name: u.name,
      role: u.role,
      createdAt: u.createdAt,
      lastLogin: u.lastLogin,
    }));
    res.json(safeUsers);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/users/:id - Lấy user theo ID
app.get("/api/users/:id", async (req, res) => {
  try {
    const user = await usersAPI.getById(Number(req.params.id));
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/users/email/:email - Lấy user theo email
app.get("/api/users/email/:email", async (req, res) => {
  try {
    const user = await usersAPI.getByEmail(req.params.email);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/users - Tạo user mới
app.post("/api/users", async (req, res) => {
  try {
    const creator = req.headers["x-user"] || "Admin";
    const user = await usersAPI.create(req.body, creator);
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PUT /api/users/:id - Cập nhật user
app.put("/api/users/:id", async (req, res) => {
  try {
    const updater = req.headers["x-user"] || "Admin";
    const user = await usersAPI.update(
      Number(req.params.id),
      req.body,
      updater
    );
    res.json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// PATCH /api/users/:id - Cập nhật một phần user
app.patch("/api/users/:id", async (req, res) => {
  try {
    const updater = req.headers["x-user"] || "Admin";
    const user = await usersAPI.update(
      Number(req.params.id),
      req.body,
      updater
    );
    res.json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// DELETE /api/users/:id - Xóa user
app.delete("/api/users/:id", async (req, res) => {
  try {
    const deleter = req.headers["x-user"] || "Admin";
    await usersAPI.delete(Number(req.params.id), deleter);
    res.json({ success: true, message: "User deleted" });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// POST /api/users/login - Update last login
app.post("/api/users/login", async (req, res) => {
  try {
    const { email } = req.body;
    const user = await usersAPI.updateLastLogin(email);
    res.json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// ============ AUDIT LOGS ENDPOINTS ============

// GET /api/audit-logs - Lấy audit logs
app.get("/api/audit-logs", async (req, res) => {
  try {
    const { limit, user, action } = req.query;
    let logs;

    if (user) {
      logs = await auditLogsAPI.getByUser(user, Number(limit) || 50);
    } else if (action) {
      logs = await auditLogsAPI.getByAction(action, Number(limit) || 50);
    } else {
      logs = await auditLogsAPI.getAll(Number(limit) || 100);
    }

    res.json(logs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/audit-logs - Ghi log
app.post("/api/audit-logs", async (req, res) => {
  try {
    const { action, user, metadata } = req.body;
    await auditLogsAPI.log(action, user, metadata);
    res.status(201).json({ success: true, message: "Log created" });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// ============ STATISTICS ENDPOINTS ============

// GET /api/stats/dashboard - Thống kê tổng quan
app.get("/api/stats/dashboard", async (req, res) => {
  try {
    const stats = await statsAPI.getDashboard();
    res.json(stats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/stats/revenue-by-month - Doanh thu theo tháng
app.get("/api/stats/revenue-by-month", async (req, res) => {
  try {
    const year = Number(req.query.year) || new Date().getFullYear();
    const revenue = await statsAPI.getRevenueByMonth(year);
    res.json(revenue);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/stats/top-products - Top sản phẩm
app.get("/api/stats/top-products", async (req, res) => {
  try {
    const limit = Number(req.query.limit) || 10;
    const topProducts = await statsAPI.getTopProducts(limit);
    res.json(topProducts);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============ HEALTH CHECK ============

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// ============ RECOMMENDATION ENDPOINTS ============

// GET /api/recommendations - Get recommendations for a user
app.get("/api/recommendations", async (req, res) => {
  try {
    const userId = req.query.userId;

    if (!userId) {
      return res.status(400).json({
        error: "userId is required",
      });
    }

    // Get recommendation service
    const recService = getRecommendationService();

    // Load data - use ML orders for better recommendations
    const orders = await loadMLOrders();
    const products = await productsAPI.getAll();

    // Get recommendations
    const recommendations = await recService.getRecommendations(
      userId,
      orders,
      products,
      { limit: 12 }
    );

    res.json(recommendations);
  } catch (error) {
    console.error("Recommendation error:", error);
    res.status(500).json({
      error: "Failed to generate recommendations",
      message: error.message,
    });
  }
});

// POST /api/recommendations/retrain - Manual retrain
app.post("/api/recommendations/retrain", async (req, res) => {
  try {
    const recService = getRecommendationService();

    const orders = await loadMLOrders();
    const products = await productsAPI.getAll();

    await recService.retrain(orders, products);

    res.json({
      message: "Retrain completed",
      status: recService.getStatus(),
    });
  } catch (error) {
    console.error("Retrain error:", error);
    res.status(500).json({
      error: "Retrain failed",
      message: error.message,
    });
  }
});

// GET /api/recommendations/status - Status endpoint
app.get("/api/recommendations/status", (req, res) => {
  try {
    const recService = getRecommendationService();
    res.json(recService.getStatus());
  } catch (error) {
    res.status(500).json({
      error: "Failed to get status",
      message: error.message,
    });
  }
});

// Root endpoint
app.get("/", (req, res) => {
  res.json({
    message: "Vựa Vui Vẻ API Server",
    version: "1.0.0",
    endpoints: {
      products: "/api/products",
      orders: "/api/orders",
      users: "/api/users",
      auditLogs: "/api/audit-logs",
      stats: "/api/stats",
      health: "/health",
    },
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: "Endpoint not found" });
});

// Error handler
app.use((err, req, res, next) => {
  console.error("Server Error:", err);
  res.status(500).json({ error: "Internal server error" });
});

// Start server
app.listen(PORT, async () => {
  console.log(`\n🚀 Vựa Vui Vẻ API Server đang chạy!`);
  console.log(`📍 URL: http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`📦 Products: http://localhost:${PORT}/api/products`);
  console.log(`📋 Orders: http://localhost:${PORT}/api/orders`);
  console.log(`👥 Users: http://localhost:${PORT}/api/users`);
  console.log(`📝 Audit Logs: http://localhost:${PORT}/api/audit-logs`);
  console.log(`📈 Stats: http://localhost:${PORT}/api/stats/dashboard`);
  console.log(
    `🎯 Recommendations: http://localhost:${PORT}/api/recommendations`
  );
  console.log(`\n✨ Server sẵn sàng nhận requests!\n`);

  // Initialize recommendation service in background
  try {
    console.log("🚀 Initializing recommendation service...");
    const recService = getRecommendationService();
    const orders = await loadMLOrders();
    const products = await productsAPI.getAll();

    console.log(`📊 Loaded ${orders.length} orders for ML training`);

    recService.initialize(orders, products).catch((err) => {
      console.error("Failed to initialize recommendation service:", err);
    });
  } catch (error) {
    console.error("Startup error:", error);
  }
});

export default app;
