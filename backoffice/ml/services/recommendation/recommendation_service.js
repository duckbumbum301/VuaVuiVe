/**
 * Recommendation Service
 * Kết hợp Apriori + Matrix Factorization + Trending
 */

import AprioriEngine from "./apriori_engine.js";
import MatrixFactorizationEngine from "./matrix_factorization.js";
import { promises as fs } from "fs";
import path from "path";

class RecommendationService {
  constructor() {
    this.apriori = new AprioriEngine({
      minSupport: 0.03,
      minConfidence: 0.2,
      minLift: 1.2,
    });

    this.mf = new MatrixFactorizationEngine({
      nFactors: 20,
      iterations: 50,
    });

    this.isInitialized = false;
    this.isTraining = false;
  }

  /**
   * Initialize service (load cache or train)
   */
  async initialize(orders, products) {
    if (this.isInitialized) return;

    console.log("🚀 Initializing Recommendation Service...");

    // Try to load from cache
    const aprioriLoaded = await this.apriori.loadRulesFromCache();
    const mfLoaded = await this.mf.loadModelFromCache();

    // If cache missing or old, retrain
    if (!aprioriLoaded || !mfLoaded) {
      await this.trainModels(orders, products);
    }

    this.isInitialized = true;
    console.log("✅ Recommendation Service initialized");
  }

  /**
   * Train both models
   */
  async trainModels(orders, products) {
    if (this.isTraining) {
      console.log("⏳ Training already in progress...");
      return;
    }

    this.isTraining = true;
    console.log("🎓 Training recommendation models...");

    try {
      // Filter valid orders
      const validOrders = orders.filter(
        (o) => o.items && o.items.length > 0 && o.status === "completed" // Only use completed orders
      );

      console.log(`📦 Training with ${validOrders.length} orders`);

      // Train in parallel
      await Promise.all([
        this.apriori.train(validOrders),
        this.mf.train(validOrders, products),
      ]);

      console.log("✅ Training completed");
    } catch (error) {
      console.error("❌ Training failed:", error);
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Get recommendations for a user
   */
  async getRecommendations(userId, orders, products, options = {}) {
    const limit = options.limit || 12;

    // Ensure initialized
    if (!this.isInitialized) {
      await this.initialize(orders, products);
    }

    // Get user's purchase history
    const userOrders = orders.filter((o) => o.user_id === userId);
    const userPurchasedIds = this.getUserPurchasedIds(userOrders);

    // Generate user signals (preferences)
    const userSignals = this.analyzeUserPreferences(userOrders, products);

    // Get recommendations from both engines
    const personalRecs = this.mf.predictForUser(userId, products, limit);
    const similarRecs = this.apriori.recommendSimilar(
      userPurchasedIds,
      products,
      limit
    );

    // Filter out already purchased items
    const purchased = new Set(userPurchasedIds.map((id) => String(id)));
    const filterPurchased = (recs) =>
      recs.filter((p) => !purchased.has(String(p.id)));

    // Get trending products (fallback)
    const trendingRecs = this.getTrendingProducts(products, limit);

    return {
      personal: filterPurchased(personalRecs),
      similar: filterPurchased(similarRecs),
      trending: filterPurchased(trendingRecs),
      userSignals: {
        topCategories: userSignals.topCategories.slice(0, 3),
        purchaseCount: userOrders.length,
        hasHistory: userOrders.length > 0,
      },
      metadata: {
        generatedAt: new Date().toISOString(),
        aprioriStats: this.apriori.getStats(),
        mfStats: this.mf.getStats(),
      },
    };
  }

  /**
   * Extract all product IDs user has purchased
   */
  getUserPurchasedIds(userOrders) {
    const ids = new Set();
    userOrders.forEach((order) => {
      order.items.forEach((item) => {
        ids.add(String(item.product_id));
      });
    });
    return Array.from(ids);
  }

  /**
   * Analyze user preferences from purchase history
   */
  analyzeUserPreferences(userOrders, products) {
    const categoryCount = {};
    const subcategoryCount = {};

    userOrders.forEach((order) => {
      order.items.forEach((item) => {
        const product = products.find((p) => p.id === item.product_id);
        if (product) {
          categoryCount[product.category] =
            (categoryCount[product.category] || 0) + 1;
          if (product.subcategory) {
            subcategoryCount[product.subcategory] =
              (subcategoryCount[product.subcategory] || 0) + 1;
          }
        }
      });
    });

    // Sort by count
    const topCategories = Object.entries(categoryCount)
      .sort((a, b) => b[1] - a[1])
      .map(([cat, count]) => cat);

    const topSubcategories = Object.entries(subcategoryCount)
      .sort((a, b) => b[1] - a[1])
      .map(([sub, count]) => sub);

    return {
      topCategories,
      topSubcategories,
      categoryCount,
      subcategoryCount,
    };
  }

  /**
   * Get trending products (popularity-based)
   */
  getTrendingProducts(products, limit = 12) {
    // Sort by addToCart count
    const trending = [...products]
      .filter((p) => p.stock > 0) // In stock only
      .sort((a, b) => (b.addToCart || 0) - (a.addToCart || 0))
      .slice(0, limit)
      .map((p) => ({
        ...p,
        recommendationReason: "Sản phẩm phổ biến tại Vựa Vui Vẻ",
      }));

    return trending;
  }

  /**
   * Manual retrain trigger (for admin)
   */
  async retrain(orders, products) {
    console.log("🔄 Manual retrain triggered");
    await this.trainModels(orders, products);
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      isInitialized: this.isInitialized,
      isTraining: this.isTraining,
      apriori: this.apriori.getStats(),
      matrixFactorization: this.mf.getStats(),
    };
  }
}

// Singleton instance
let instance = null;

function getRecommendationService() {
  if (!instance) {
    instance = new RecommendationService();
  }
  return instance;
}

export { RecommendationService, getRecommendationService };
