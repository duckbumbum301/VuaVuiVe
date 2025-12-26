/**
 * Script chuẩn bị data cho ML model
 * Chuyển đổi format orders và tạo sample data nếu cần
 */

import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATA_DIR = path.join(__dirname, "../../data");

/**
 * Load JSON file
 */
async function loadJSON(filePath) {
  try {
    const data = await fs.readFile(filePath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error loading ${filePath}:`, error.message);
    return [];
  }
}

/**
 * Save JSON file
 */
async function saveJSON(filePath, data) {
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
  console.log(`✅ Saved: ${filePath}`);
}

/**
 * Chuyển đổi orders sang format ML
 */
function transformOrdersForML(orders, users, products) {
  const emailToUserId = {};
  users.forEach((u) => {
    if (u.email) emailToUserId[u.email.toLowerCase()] = u.id;
  });

  const productIdMap = new Set(products.map((p) => String(p.id)));

  const transformed = orders.map((order) => {
    // Map email to user_id
    const email = (order.email || "").toLowerCase();
    const user_id = emailToUserId[email] || email;

    // Transform items: productId -> product_id
    // Handle both array and object formats
    let itemsArray = [];
    if (Array.isArray(order.items)) {
      itemsArray = order.items;
    } else if (order.items && typeof order.items === "object") {
      // Convert object to array
      itemsArray = Object.entries(order.items).map(([id, qty]) => ({
        productId: id,
        product_id: id,
        quantity: qty,
      }));
    }

    const items = itemsArray
      .filter((item) => {
        const pid = String(item.productId || item.product_id);
        return productIdMap.has(pid);
      })
      .map((item) => ({
        product_id: String(item.productId || item.product_id),
        quantity: item.quantity || 1,
        price: item.price || 0,
      }));

    // Determine status
    let status = order.status || "pending";
    if (
      order.delivery_status === "delivered" ||
      status === "delivered" ||
      status === "completed"
    ) {
      status = "completed";
    }

    return {
      id: order.id,
      user_id: user_id,
      items: items,
      totalAmount: order.totalAmount || 0,
      status: status,
      createdAt: order.createdAt || new Date().toISOString(),
    };
  });

  return transformed.filter((o) => o.items.length > 0);
}

/**
 * Generate sample users (nếu cần)
 */
function generateSampleUsers(existingUsers, count = 10) {
  const names = [
    "Nguyễn Văn",
    "Trần Thị",
    "Lê Văn",
    "Phạm Thị",
    "Hoàng Văn",
    "Vũ Thị",
    "Đặng Văn",
    "Bùi Thị",
    "Đỗ Văn",
    "Ngô Thị",
  ];
  const suffixes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

  const maxId = Math.max(...existingUsers.map((u) => u.id || 0));
  const newUsers = [];

  for (let i = 0; i < count; i++) {
    const name = `${names[i % names.length]} ${suffixes[i % suffixes.length]}`;
    const email = `customer${maxId + i + 1}@gmail.com`;

    newUsers.push({
      id: maxId + i + 1,
      email: email,
      name: name,
      role: "Customer",
      createdAt: new Date().toISOString(),
    });
  }

  return newUsers;
}

/**
 * Generate sample orders (nếu cần)
 */
function generateSampleOrders(users, products, count = 30) {
  const sampleOrders = [];
  const customerUsers = users.filter((u) => u.role === "Customer");

  for (let i = 0; i < count; i++) {
    const user = customerUsers[i % customerUsers.length];
    const numItems = Math.floor(Math.random() * 4) + 2; // 2-5 items
    const selectedProducts = [];

    // Random select products
    for (let j = 0; j < numItems; j++) {
      const product = products[Math.floor(Math.random() * products.length)];
      selectedProducts.push({
        product_id: String(product.id),
        quantity: Math.floor(Math.random() * 3) + 1,
        price: product.price || 10000,
      });
    }

    const totalAmount = selectedProducts.reduce(
      (sum, item) => sum + item.quantity * item.price,
      0
    );

    sampleOrders.push({
      id: `ORD2025SAMPLE${String(i + 1).padStart(3, "0")}`,
      user_id: user.id,
      items: selectedProducts,
      totalAmount: totalAmount,
      status: "completed",
      createdAt: new Date(
        Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
      ).toISOString(),
    });
  }

  return sampleOrders;
}

/**
 * Main function
 */
async function main() {
  console.log("🚀 Bắt đầu chuẩn bị data cho ML...\n");

  // Load existing data
  const orders = await loadJSON(path.join(DATA_DIR, "orders.json"));
  const users = await loadJSON(path.join(DATA_DIR, "users.json"));
  const products = await loadJSON(path.join(DATA_DIR, "products.json"));

  console.log("📊 Data hiện tại:");
  console.log(`   - Orders: ${orders.length}`);
  console.log(`   - Users: ${users.length}`);
  console.log(`   - Products: ${products.length}\n`);

  // Step 1: Generate more users if needed
  let allUsers = [...users];
  if (users.length < 15) {
    console.log("📝 Tạo thêm sample users...");
    const newUsers = generateSampleUsers(users, 15 - users.length);
    allUsers = [...users, ...newUsers];
    await saveJSON(path.join(DATA_DIR, "users.json"), allUsers);
    console.log(`   ✅ Đã thêm ${newUsers.length} users mới\n`);
  }

  // Step 2: Transform existing orders
  console.log("🔄 Chuyển đổi format orders...");
  const transformedOrders = transformOrdersForML(orders, allUsers, products);
  console.log(`   ✅ Đã transform ${transformedOrders.length} orders\n`);

  // Step 3: Generate sample orders if needed
  let allOrders = [...transformedOrders];
  const completedOrders = transformedOrders.filter(
    (o) => o.status === "completed"
  ).length;

  // Increase target to 100 orders for better ML training
  const targetOrders = 100;
  if (completedOrders < targetOrders) {
    console.log("📝 Tạo thêm sample orders để train...");
    const customerUsers = allUsers.filter((u) => u.role === "Customer");
    const sampleOrders = generateSampleOrders(
      customerUsers,
      products,
      targetOrders - completedOrders
    );
    allOrders = [...transformedOrders, ...sampleOrders];
    console.log(`   ✅ Đã thêm ${sampleOrders.length} orders mới\n`);
  }

  // Step 4: Save ML-ready orders
  const mlOrdersPath = path.join(DATA_DIR, "orders-ml.json");
  await saveJSON(mlOrdersPath, allOrders);

  // Step 5: Statistics
  const completedCount = allOrders.filter(
    (o) => o.status === "completed"
  ).length;
  const uniqueUsers = new Set(allOrders.map((o) => o.user_id)).size;
  const uniqueProducts = new Set(
    allOrders.flatMap((o) => o.items.map((i) => i.product_id))
  ).size;

  console.log("📈 Thống kê data ML:");
  console.log(`   - Total orders: ${allOrders.length}`);
  console.log(`   - Completed orders: ${completedCount}`);
  console.log(`   - Unique users: ${uniqueUsers}`);
  console.log(`   - Unique products in orders: ${uniqueProducts}`);
  console.log(`   - Total users: ${allUsers.length}`);
  console.log(`   - Total products: ${products.length}\n`);

  // Step 6: Validation
  console.log("✅ Kiểm tra data quality:");

  if (completedCount >= 30) {
    console.log("   ✅ Đủ orders để train Apriori (cần ≥30)");
  } else {
    console.log(`   ⚠️  Chưa đủ orders (${completedCount}/30), nên tăng data`);
  }

  if (uniqueUsers >= 10) {
    console.log("   ✅ Đủ users để train Matrix Factorization (cần ≥10)");
  } else {
    console.log(`   ⚠️  Chưa đủ users (${uniqueUsers}/10), nên tăng users`);
  }

  if (uniqueProducts >= 30) {
    console.log("   ✅ Đủ products có trong orders (cần ≥30)");
  } else {
    console.log(
      `   ⚠️  Chưa đủ products (${uniqueProducts}/30), nên đa dạng orders`
    );
  }

  console.log("\n🎉 Hoàn tất! Data đã sẵn sàng tại: orders-ml.json");
  console.log(
    "\n📝 Tiếp theo: Cập nhật server.js để dùng orders-ml.json thay vì orders.json"
  );
}

main().catch(console.error);
