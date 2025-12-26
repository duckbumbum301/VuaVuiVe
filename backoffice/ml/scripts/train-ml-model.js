/**
 * Script train ML model
 * Chạy script này để train Apriori + Matrix Factorization
 */

import { getRecommendationService } from "../services/recommendation/recommendation_service.js";
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
  const data = await fs.readFile(filePath, "utf-8");
  return JSON.parse(data);
}

/**
 * Main training function
 */
async function main() {
  console.log("🚀 Bắt đầu train ML models...\n");

  try {
    // Load data
    console.log("📂 Đang load data...");
    const orders = await loadJSON(path.join(DATA_DIR, "orders-ml.json"));
    const products = await loadJSON(path.join(DATA_DIR, "products.json"));

    console.log(`   ✅ Loaded ${orders.length} orders`);
    console.log(`   ✅ Loaded ${products.length} products\n`);

    // Filter completed orders
    const completedOrders = orders.filter((o) => o.status === "completed");
    console.log(
      `📊 Training with ${completedOrders.length} completed orders\n`
    );

    // Initialize recommendation service
    console.log("🤖 Initializing recommendation service...");
    const recService = getRecommendationService();

    // Train models
    console.log("🎓 Training models...\n");
    await recService.trainModels(completedOrders, products);

    // Get statistics
    console.log("\n📈 Kết quả training:");
    const status = recService.getStatus();

    console.log("\n🔹 Apriori Algorithm:");
    console.log(`   - Total rules: ${status.apriori.totalRules}`);
    console.log(`   - Last updated: ${status.apriori.lastUpdated}`);
    console.log(
      `   - Config: support=${status.apriori.config.minSupport}, confidence=${status.apriori.config.minConfidence}, lift=${status.apriori.config.minLift}`
    );

    console.log("\n🔹 Matrix Factorization:");
    console.log(`   - Users: ${status.matrixFactorization.numUsers}`);
    console.log(`   - Items: ${status.matrixFactorization.numItems}`);
    console.log(`   - Factors: ${status.matrixFactorization.nFactors}`);
    console.log(`   - Last updated: ${status.matrixFactorization.lastUpdated}`);

    // Test recommendations for a sample user
    if (completedOrders.length > 0) {
      const sampleOrder = completedOrders[0];
      const userId = sampleOrder.user_id;

      console.log(`\n🧪 Testing recommendations for user: ${userId}`);
      const recommendations = await recService.getRecommendations(
        userId,
        orders,
        products,
        { limit: 5 }
      );

      console.log(`   - Personal: ${recommendations.personal.length} products`);
      console.log(`   - Similar: ${recommendations.similar.length} products`);
      console.log(`   - Trending: ${recommendations.trending.length} products`);

      if (recommendations.personal.length > 0) {
        console.log("\n   Top 3 personal recommendations:");
        recommendations.personal.slice(0, 3).forEach((p, i) => {
          console.log(`      ${i + 1}. ${p.name} - ${p.recommendationReason}`);
        });
      }
    }

    console.log("\n✅ Training hoàn tất!");
    console.log(
      "💾 Model đã được cache tại: backoffice/ml/services/recommendation/recommendation_cache.json"
    );
    console.log("\n🚀 Khởi động server để sử dụng: node backoffice/server.js");
  } catch (error) {
    console.error("\n❌ Training failed:", error);
    throw error;
  }
}

main().catch(console.error);
