/**
 * ML Model Evaluation Script
 * Đánh giá chất lượng recommendations
 */

import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import recommendation service
import { getRecommendationService } from "../services/recommendation/recommendation_service.js";

/**
 * Load JSON file
 */
async function loadJSON(filePath) {
  const data = await fs.readFile(filePath, "utf-8");
  return JSON.parse(data);
}

/**
 * Calculate Precision@K
 * Percentage of recommended items that are relevant
 */
function precisionAtK(recommended, relevant, k = 5) {
  const topK = recommended.slice(0, k);
  const relevantSet = new Set(relevant);
  const hits = topK.filter((item) => relevantSet.has(item)).length;
  return hits / k;
}

/**
 * Calculate Recall@K
 * Percentage of relevant items that are recommended
 */
function recallAtK(recommended, relevant, k = 5) {
  if (relevant.length === 0) return 0;
  const topK = recommended.slice(0, k);
  const relevantSet = new Set(relevant);
  const hits = topK.filter((item) => relevantSet.has(item)).length;
  return hits / relevant.length;
}

/**
 * Calculate Coverage
 * Percentage of products that appear in recommendations
 */
function calculateCoverage(allRecommendations, totalProducts) {
  const recommendedProducts = new Set();
  allRecommendations.forEach((recs) => {
    recs.forEach((productId) => recommendedProducts.add(productId));
  });
  return recommendedProducts.size / totalProducts;
}

/**
 * Calculate Diversity
 * Average pairwise difference between recommendations
 */
function calculateDiversity(recommendations, products) {
  if (recommendations.length < 2) return 1.0;

  const productMap = new Map(products.map((p) => [String(p.id), p]));
  let totalDistance = 0;
  let pairs = 0;

  for (let i = 0; i < recommendations.length - 1; i++) {
    for (let j = i + 1; j < recommendations.length; j++) {
      const p1 = productMap.get(recommendations[i]);
      const p2 = productMap.get(recommendations[j]);

      if (p1 && p2) {
        // Simple diversity: different categories = 1, same = 0
        const distance =
          p1.category !== p2.category || p1.subcategory !== p2.subcategory
            ? 1
            : 0;
        totalDistance += distance;
        pairs++;
      }
    }
  }

  return pairs > 0 ? totalDistance / pairs : 0;
}

/**
 * Main evaluation function
 */
async function evaluateModel() {
  console.log("🔍 Bắt đầu đánh giá ML Model...\n");

  // Load data
  const ordersPath = path.join(__dirname, "../../data", "orders-ml.json");
  const productsPath = path.join(__dirname, "../../data", "products.json");
  const orders = await loadJSON(ordersPath);
  const products = await loadJSON(productsPath);

  const completedOrders = orders.filter((o) => o.status === "completed");
  console.log(
    `📊 Data: ${completedOrders.length} orders, ${products.length} products\n`
  );

  // Initialize recommendation service
  const recService = getRecommendationService();
  await recService.initialize(completedOrders, products);

  console.log("✅ Model initialized\n");

  // Test recommendations for sample users
  const userIds = [...new Set(completedOrders.map((o) => o.user_id))];
  const sampleUsers = userIds.slice(0, Math.min(10, userIds.length));

  console.log(
    `🎯 Testing recommendations for ${sampleUsers.length} users...\n`
  );

  let totalPrecision = 0;
  let totalRecall = 0;
  let totalDiversity = 0;
  let usersWithRecs = 0;
  const allRecommendations = [];

  for (const userId of sampleUsers) {
    // Get user's purchase history
    const userOrders = completedOrders.filter((o) => o.user_id === userId);
    const purchasedProducts = new Set();
    userOrders.forEach((order) => {
      order.items.forEach((item) => purchasedProducts.add(item.product_id));
    });

    // Get recommendations
    const result = await recService.getRecommendations(
      userId,
      completedOrders,
      products,
      { limit: 10 }
    );
    const recommendations = [
      ...(result.personal || []),
      ...(result.similar || []),
      ...(result.trending || []),
    ]
      .map((r) => String(r.id))
      .slice(0, 10);

    if (recommendations.length > 0) {
      allRecommendations.push(recommendations);
      usersWithRecs++;

      // Calculate diversity
      const diversity = calculateDiversity(recommendations, products);
      totalDiversity += diversity;

      console.log(`User ${userId}:`);
      console.log(`  📚 Purchased: ${purchasedProducts.size} products`);
      console.log(`  💡 Recommended: ${recommendations.length} products`);
      console.log(`  🎲 Diversity: ${(diversity * 100).toFixed(1)}%`);

      // Show sample recommendations
      const sampleRecs = recommendations.slice(0, 3);
      const recProducts = sampleRecs
        .map((id) => products.find((p) => String(p.id) === id))
        .filter(Boolean);
      console.log(
        `  🏷️  Top 3: ${recProducts.map((p) => p.name).join(", ")}\n`
      );
    }
  }

  // Calculate overall metrics
  const coverage = calculateCoverage(allRecommendations, products.length);
  const avgDiversity = usersWithRecs > 0 ? totalDiversity / usersWithRecs : 0;

  console.log("\n" + "=".repeat(60));
  console.log("📈 OVERALL METRICS:");
  console.log("=".repeat(60));
  console.log(
    `Users with recommendations: ${usersWithRecs}/${sampleUsers.length}`
  );
  console.log(
    `Coverage: ${(coverage * 100).toFixed(1)}% (${Math.floor(
      coverage * products.length
    )}/${products.length} products)`
  );
  console.log(`Average Diversity: ${(avgDiversity * 100).toFixed(1)}%`);
  console.log("=".repeat(60) + "\n");

  // Check Apriori rules
  console.log("🔗 Apriori Association Rules:");
  const aprioriEngine = recService.apriori;
  if (aprioriEngine && aprioriEngine.rules && aprioriEngine.rules.length > 0) {
    console.log(`   ✅ ${aprioriEngine.rules.length} rules found`);
    // Show top 5 rules
    const topRules = aprioriEngine.rules.slice(0, 5);
    topRules.forEach((rule, i) => {
      console.log(
        `   ${i + 1}. [${rule.antecedent.join(",")}] → [${rule.consequent.join(
          ","
        )}]`
      );
      console.log(
        `      Confidence: ${(rule.confidence * 100).toFixed(
          1
        )}%, Lift: ${rule.lift.toFixed(2)}`
      );
    });
  } else {
    console.log("   ⚠️  No rules found - thresholds may be too high");
  }

  console.log("\n✨ Evaluation completed!");
}

// Run evaluation
evaluateModel().catch((err) => {
  console.error("❌ Error:", err);
  process.exit(1);
});
