/**
 * Apriori Algorithm Engine
 * Tìm association rules từ lịch sử đơn hàng
 */

import Apriori from "apriori";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class AprioriEngine {
  constructor(config = {}) {
    // Lowered thresholds for better rule generation with limited data
    this.minSupport = config.minSupport || 0.02; // 2% transactions (was 3%)
    this.minConfidence = config.minConfidence || 0.15; // 15% confidence (was 20%)
    this.minLift = config.minLift || 1.0; // Lift > 1.0 (was 1.2) - any positive correlation
    this.rules = [];
    this.lastUpdated = null;
  }

  /**
   * STEP 1: Chuyển đổi orders thành transaction format
   * Input: [{user_id, items: [{product_id, quantity}]}]
   * Output: [['P1', 'P2'], ['P1', 'P3'], ...]
   */
  buildTransactionDatabase(orders) {
    const transactions = orders.map((order) => {
      // Lấy unique product IDs từ mỗi order
      const productIds = order.items.map((item) => String(item.product_id));
      return [...new Set(productIds)]; // Remove duplicates
    });

    console.log(`📦 Built transaction DB: ${transactions.length} transactions`);
    return transactions;
  }

  /**
   * STEP 2: Chạy Apriori algorithm để tìm frequent itemsets & rules
   */
  async train(orders) {
    if (!orders || orders.length === 0) {
      console.warn("⚠️ No orders to train Apriori");
      return;
    }

    console.log("🔄 Training Apriori algorithm...");
    const transactions = this.buildTransactionDatabase(orders);

    // Run Apriori algorithm
    const apriori = new Apriori.Algorithm(
      this.minSupport,
      this.minConfidence,
      true // Enable frequent itemset mining
    );

    // Execute algorithm
    const result = apriori.analyze(transactions);

    // Extract association rules
    // Note: Apriori library doesn't provide lift in associationRules,
    // so we calculate a pseudo-lift score based on confidence
    this.rules = result.associationRules.map((rule) => ({
      antecedent: rule.lhs, // Left hand side (IF)
      consequent: rule.rhs, // Right hand side (THEN)
      support: rule.support,
      confidence: rule.confidence,
      lift: rule.lift || 1.0, // Library doesn't provide lift, default to 1.0
      score: rule.confidence, // Use confidence as primary score
    }));

    // Filter by confidence threshold (since lift is not reliable)
    // Only keep rules with meaningful confidence
    this.rules = this.rules.filter((r) => r.confidence >= this.minConfidence);

    // Sort by confidence (descending)
    this.rules.sort((a, b) => b.confidence - a.confidence);

    this.lastUpdated = new Date().toISOString();
    console.log(`✅ Apriori trained: ${this.rules.length} rules found`);

    // Cache rules
    await this.saveRulesToCache();
  }

  /**
   * STEP 3: Gợi ý sản phẩm dựa trên items user đã mua
   */
  recommendSimilar(userPurchasedIds, allProducts, limit = 12) {
    if (!userPurchasedIds || userPurchasedIds.length === 0) {
      return [];
    }

    const purchased = new Set(userPurchasedIds.map((id) => String(id)));
    const recommendations = new Map(); // product_id -> {score, reasons[]}

    // Tìm rules có antecedent chứa sản phẩm user đã mua
    for (const rule of this.rules) {
      const hasMatch = rule.antecedent.some((item) => purchased.has(item));

      if (hasMatch) {
        // Add all consequent items
        for (const itemId of rule.consequent) {
          if (!purchased.has(itemId)) {
            if (!recommendations.has(itemId)) {
              recommendations.set(itemId, {
                product_id: itemId,
                score: 0,
                reasons: [],
              });
            }

            const rec = recommendations.get(itemId);
            rec.score += rule.score;
            rec.reasons.push({
              type: "apriori",
              confidence: rule.confidence,
              lift: rule.lift,
              fromItems: rule.antecedent,
            });
          }
        }
      }
    }

    // Convert to array và sort
    let results = Array.from(recommendations.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    // Enrich với product details
    results = results
      .map((rec) => {
        const product = allProducts.find(
          (p) => String(p.id) === rec.product_id
        );
        return {
          ...product,
          recommendationScore: rec.score,
          recommendationReason: "Thường mua cùng các sản phẩm bạn đã chọn",
        };
      })
      .filter((p) => p.id); // Remove null products

    return results;
  }

  /**
   * STEP 4: Cache rules to file
   */
  async saveRulesToCache() {
    const cachePath = path.join(__dirname, "recommendation_cache.json");
    let cache = {};

    // Load existing cache
    try {
      const data = await fs.readFile(cachePath, "utf-8");
      cache = JSON.parse(data);
    } catch (error) {
      // File doesn't exist yet
    }

    // Update apriori section
    cache.apriori = {
      rules: this.rules,
      lastUpdated: this.lastUpdated,
      config: {
        minSupport: this.minSupport,
        minConfidence: this.minConfidence,
        minLift: this.minLift,
      },
    };

    try {
      await fs.writeFile(cachePath, JSON.stringify(cache, null, 2));
      console.log("💾 Apriori rules cached");
    } catch (error) {
      console.error("❌ Failed to cache rules:", error);
    }
  }

  /**
   * Load rules from cache
   */
  async loadRulesFromCache() {
    const cachePath = path.join(__dirname, "recommendation_cache.json");

    try {
      const data = await fs.readFile(cachePath, "utf-8");
      const cache = JSON.parse(data);

      if (cache.apriori) {
        this.rules = cache.apriori.rules || [];
        this.lastUpdated = cache.apriori.lastUpdated;
        console.log(`📂 Loaded ${this.rules.length} cached Apriori rules`);
        return true;
      }
    } catch (error) {
      console.log("ℹ️ No cached rules found, will train on first request");
    }

    return false;
  }

  /**
   * Get rule statistics
   */
  getStats() {
    return {
      totalRules: this.rules.length,
      lastUpdated: this.lastUpdated,
      config: {
        minSupport: this.minSupport,
        minConfidence: this.minConfidence,
        minLift: this.minLift,
      },
    };
  }
}

export default AprioriEngine;
