# Hướng dẫn Triển khai Hệ thống Recommendation với Apriori & Matrix Factorization

## 📋 Tổng quan

Tài liệu này hướng dẫn chi tiết từng bước để xây dựng hệ thống recommendation cho trang `recommended.html` sử dụng:

- **Apriori Algorithm** (cho "Sản phẩm tương tự")
- **Matrix Factorization** (cho "Gợi ý cá nhân")

---

## 🎯 Mục tiêu

- Section 1: **Gợi ý cá nhân** → Sử dụng Matrix Factorization
- Section 2: **Sản phẩm tương tự** → Sử dụng Apriori Algorithm
- Section 3: **Xu hướng** → Popularity-based (đã có sẵn)

---

## 🆚 So sánh với SEASON PAGE

| Tính năng       | Recommended Page       | Season Page                          |
| --------------- | ---------------------- | ------------------------------------ |
| **Mục đích**    | **Cá nhân hóa cao**    | Phù hợp với thời điểm                |
| **Input chính** | **User history**       | Season + Weather + Date              |
| **CF**          | **Standard MF (50%)**  | Seasonal CF (30%)                    |
| **Apriori**     | **Global rules (30%)** | S-Apriori với seasonal weights (30%) |
| **Content**     | **20%**                | -                                    |
| **Trend**       | -                      | Time-series analysis (25%)           |
| **Weather**     | -                      | Weather-aware (15%)                  |
| **Re-train**    | **Weekly**             | Daily (seasonal patterns thay đổi)   |
| **Best for**    | **Returning users**    | All users + seasonal context         |

**Kết luận:**

- **Recommended Page** = Personalization-first
- **Season Page** = Context-first (seasonal + weather)

---

## 📁 Cấu trúc File cần tạo

```
Group5_FinalProject/
├── backoffice/
│   ├── ml/
│   │   ├── models/
│   │   │   ├── SeasonalRecommender.js
│   │   │   └── TimeAwareRecommender.js
│   │   ├── scripts/
│   │   │   ├── train-ml-model.js
│   │   │   ├── evaluate-ml-model.js
│   │   │   └── prepare-ml-data.js
│   │   └── services/
│   │       └── recommendation/
│   │           ├── apriori_engine.js          [MỚI]
│   │           ├── matrix_factorization.js    [MỚI]
│   │           ├── recommendation_service.js  [MỚI]
│   │           └── recommendation_cache.json  [MỚI - auto generated]
│   └── server.js                      [CẬP NHẬT]
├── js/
│   └── recommended.js                 [CẬP NHẬT]
└── RECOMMENDATION_IMPLEMENTATION.md   [File này]
```

---

## 🔧 BƯỚC 1: Cài đặt Dependencies

### 1.1. Cài đặt thư viện ML cho Node.js

```bash
cd backoffice
npm install ml-matrix ml-pca apriori mathjs
```

### 1.2. Giải thích các thư viện

| Thư viện    | Mục đích                         |
| ----------- | -------------------------------- |
| `ml-matrix` | Matrix operations cho MF         |
| `ml-pca`    | PCA/SVD cho factorization        |
| `apriori`   | Apriori algorithm implementation |
| `mathjs`    | Math utilities                   |

---

## 🧠 BƯỚC 2: Xây dựng Apriori Engine

### 2.1. Tạo file `backoffice/ml/services/recommendation/apriori_engine.js`

```javascript
/**
 * Apriori Algorithm Engine
 * Tìm association rules từ lịch sử đơn hàng
 */

const Apriori = require("apriori");
const fs = require("fs").promises;
const path = require("path");

class AprioriEngine {
  constructor(config = {}) {
    this.minSupport = config.minSupport || 0.03; // 3% transactions
    this.minConfidence = config.minConfidence || 0.2; // 20% confidence
    this.minLift = config.minLift || 1.2; // Lift > 1.2
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
    this.rules = result.associationRules.map((rule) => ({
      antecedent: rule.lhs, // Left hand side (IF)
      consequent: rule.rhs, // Right hand side (THEN)
      support: rule.support,
      confidence: rule.confidence,
      lift: rule.lift || rule.confidence / rule.support,
      score: rule.confidence * (rule.lift || 1),
    }));

    // Filter by lift threshold
    this.rules = this.rules.filter((r) => r.lift >= this.minLift);

    // Sort by score
    this.rules.sort((a, b) => b.score - a.score);

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
    const cache = {
      apriori: {
        rules: this.rules,
        lastUpdated: this.lastUpdated,
        config: {
          minSupport: this.minSupport,
          minConfidence: this.minConfidence,
          minLift: this.minLift,
        },
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

module.exports = AprioriEngine;
```

---

## 🎲 BƯỚC 3: Xây dựng Matrix Factorization Engine

### 3.1. Tạo file `backoffice/ml/services/recommendation/matrix_factorization.js`

```javascript
/**
 * Matrix Factorization Engine
 * Collaborative Filtering sử dụng SVD/ALS
 */

const { Matrix, SingularValueDecomposition } = require("ml-matrix");
const fs = require("fs").promises;
const path = require("path");

class MatrixFactorizationEngine {
  constructor(config = {}) {
    this.nFactors = config.nFactors || 20; // Latent dimensions
    this.learningRate = config.learningRate || 0.01;
    this.regularization = config.regularization || 0.01;
    this.iterations = config.iterations || 50;

    this.userFactors = null; // P matrix (m x k)
    this.itemFactors = null; // Q matrix (n x k)
    this.userIds = [];
    this.itemIds = [];
    this.lastUpdated = null;
  }

  /**
   * STEP 1: Build User-Item Rating Matrix
   * R[i,j] = rating/frequency của user i cho item j
   */
  buildRatingMatrix(orders, allProducts) {
    // Extract unique users và items
    const userSet = new Set();
    const itemSet = new Set();

    orders.forEach((order) => {
      userSet.add(String(order.user_id));
      order.items.forEach((item) => {
        itemSet.add(String(item.product_id));
      });
    });

    this.userIds = Array.from(userSet).sort();
    this.itemIds = Array.from(itemSet).sort();

    const m = this.userIds.length; // num users
    const n = this.itemIds.length; // num items

    console.log(`📊 Building rating matrix: ${m} users × ${n} items`);

    // Create mapping
    const userIdx = {};
    const itemIdx = {};
    this.userIds.forEach((id, i) => (userIdx[id] = i));
    this.itemIds.forEach((id, i) => (itemIdx[id] = i));

    // Initialize matrix with zeros
    const R = Array(m)
      .fill(0)
      .map(() => Array(n).fill(0));

    // Fill matrix: rating = quantity or frequency
    orders.forEach((order) => {
      const uIdx = userIdx[String(order.user_id)];

      order.items.forEach((item) => {
        const iIdx = itemIdx[String(item.product_id)];
        if (uIdx !== undefined && iIdx !== undefined) {
          // Rating = quantity (or 1 for binary)
          R[uIdx][iIdx] += item.quantity || 1;
        }
      });
    });

    // Normalize ratings (optional: scale to 0-5)
    const maxRating = Math.max(...R.flat());
    if (maxRating > 0) {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          if (R[i][j] > 0) {
            R[i][j] = (R[i][j] / maxRating) * 5; // Scale to 0-5
          }
        }
      }
    }

    return new Matrix(R);
  }

  /**
   * STEP 2: Factorize R ≈ P × Q^T using SVD
   * Simplified approach using Singular Value Decomposition
   */
  async train(orders, allProducts) {
    if (!orders || orders.length === 0) {
      console.warn("⚠️ No orders to train Matrix Factorization");
      return;
    }

    console.log("🔄 Training Matrix Factorization (SVD)...");

    // Build rating matrix
    const R = this.buildRatingMatrix(orders, allProducts);
    const m = R.rows; // users
    const n = R.columns; // items

    try {
      // Perform SVD: R = U × Σ × V^T
      const svd = new SingularValueDecomposition(R, {
        computeLeftSingularVectors: true,
        computeRightSingularVectors: true,
      });

      // Extract top k factors
      const k = Math.min(this.nFactors, svd.diagonalMatrix.rows - 1);

      // P = U_k × Σ_k^(1/2)  (user factors)
      const U = svd.leftSingularVectors;
      const S = svd.diagonal;

      this.userFactors = Matrix.zeros(m, k);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          this.userFactors.set(i, j, U.get(i, j) * Math.sqrt(S[j]));
        }
      }

      // Q = V_k × Σ_k^(1/2)  (item factors)
      const V = svd.rightSingularVectors;

      this.itemFactors = Matrix.zeros(n, k);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
          this.itemFactors.set(i, j, V.get(i, j) * Math.sqrt(S[j]));
        }
      }

      this.lastUpdated = new Date().toISOString();
      console.log(`✅ Matrix Factorization trained: ${k} latent factors`);

      // Cache model
      await this.saveModelToCache();
    } catch (error) {
      console.error("❌ SVD failed:", error.message);
      // Fallback: Random initialization
      this.initializeRandomFactors(m, n);
    }
  }

  /**
   * Fallback: Random initialization if SVD fails
   */
  initializeRandomFactors(m, n) {
    const k = this.nFactors;
    this.userFactors = Matrix.rand(m, k).mul(0.1);
    this.itemFactors = Matrix.rand(n, k).mul(0.1);
    console.log("ℹ️ Initialized random factors");
  }

  /**
   * STEP 3: Predict ratings for a user
   * predicted_rating[j] = P[userIdx] · Q[j]^T
   */
  predictForUser(userId, allProducts, limit = 12) {
    if (!this.userFactors || !this.itemFactors) {
      console.warn("⚠️ Model not trained yet");
      return [];
    }

    const userIdx = this.userIds.indexOf(String(userId));
    if (userIdx === -1) {
      console.log(`ℹ️ User ${userId} not in training set (cold start)`);
      return [];
    }

    // Get user factor vector
    const userVector = this.userFactors.getRow(userIdx);

    // Compute predicted ratings for all items
    const predictions = [];
    for (let itemIdx = 0; itemIdx < this.itemIds.length; itemIdx++) {
      const itemVector = this.itemFactors.getRow(itemIdx);

      // Dot product
      let score = 0;
      for (let k = 0; k < userVector.length; k++) {
        score += userVector[k] * itemVector[k];
      }

      predictions.push({
        itemId: this.itemIds[itemIdx],
        score: score,
      });
    }

    // Sort by score descending
    predictions.sort((a, b) => b.score - a.score);

    // Take top N and enrich with product details
    const results = predictions
      .slice(0, limit)
      .map((pred) => {
        const product = allProducts.find((p) => String(p.id) === pred.itemId);
        if (!product) return null;

        return {
          ...product,
          recommendationScore: pred.score,
          recommendationReason: "Dựa trên sở thích cá nhân của bạn",
        };
      })
      .filter((p) => p !== null);

    return results;
  }

  /**
   * STEP 4: Cache model to file
   */
  async saveModelToCache() {
    const cachePath = path.join(__dirname, "recommendation_cache.json");

    // Load existing cache
    let cache = {};
    try {
      const data = await fs.readFile(cachePath, "utf-8");
      cache = JSON.parse(data);
    } catch (error) {
      // File doesn't exist yet
    }

    // Update MF section
    cache.matrixFactorization = {
      userFactors: this.userFactors.to2DArray(),
      itemFactors: this.itemFactors.to2DArray(),
      userIds: this.userIds,
      itemIds: this.itemIds,
      lastUpdated: this.lastUpdated,
      config: {
        nFactors: this.nFactors,
      },
    };

    try {
      await fs.writeFile(cachePath, JSON.stringify(cache, null, 2));
      console.log("💾 Matrix Factorization model cached");
    } catch (error) {
      console.error("❌ Failed to cache model:", error);
    }
  }

  /**
   * Load model from cache
   */
  async loadModelFromCache() {
    const cachePath = path.join(__dirname, "recommendation_cache.json");

    try {
      const data = await fs.readFile(cachePath, "utf-8");
      const cache = JSON.parse(data);

      if (cache.matrixFactorization) {
        const mf = cache.matrixFactorization;
        this.userFactors = new Matrix(mf.userFactors);
        this.itemFactors = new Matrix(mf.itemFactors);
        this.userIds = mf.userIds;
        this.itemIds = mf.itemIds;
        this.lastUpdated = mf.lastUpdated;

        console.log(
          `📂 Loaded cached MF model: ${this.userIds.length} users, ${this.itemIds.length} items`
        );
        return true;
      }
    } catch (error) {
      console.log("ℹ️ No cached model found, will train on first request");
    }

    return false;
  }

  /**
   * Get model statistics
   */
  getStats() {
    return {
      numUsers: this.userIds.length,
      numItems: this.itemIds.length,
      nFactors: this.nFactors,
      lastUpdated: this.lastUpdated,
    };
  }
}

module.exports = MatrixFactorizationEngine;
```

---

## 🎯 BƯỚC 4: Xây dựng Recommendation Service (Hybrid)

### 4.1. Tạo file `backoffice/ml/services/recommendation/recommendation_service.js`

```javascript
/**
 * Recommendation Service
 * Kết hợp Apriori + Matrix Factorization + Trending
 */

const AprioriEngine = require("./apriori_engine");
const MatrixFactorizationEngine = require("./matrix_factorization");
const fs = require("fs").promises;
const path = require("path");

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

module.exports = { RecommendationService, getRecommendationService };
```

---

## 🌐 BƯỚC 5: Cập nhật Backend Server

### 5.1. Sửa file `backoffice/server.js`

Thêm code sau vào file `server.js`:

```javascript
// Thêm import ở đầu file
const {
  getRecommendationService,
} = require("./recommendation/recommendation_service");

// Khởi tạo recommendation service
const recService = getRecommendationService();

// Thêm route mới cho recommendations API
app.get("/api/recommendations", async (req, res) => {
  try {
    const userId = req.query.userId;

    if (!userId) {
      return res.status(400).json({
        error: "userId is required",
      });
    }

    // Load data
    const orders = await loadJSON(path.join(__dirname, "data", "orders.json"));
    const products = await loadJSON(
      path.join(__dirname, "data", "products.json")
    );

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

// Admin endpoint: Manual retrain
app.post("/api/recommendations/retrain", async (req, res) => {
  try {
    const orders = await loadJSON(path.join(__dirname, "data", "orders.json"));
    const products = await loadJSON(
      path.join(__dirname, "data", "products.json")
    );

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

// Status endpoint
app.get("/api/recommendations/status", (req, res) => {
  res.json(recService.getStatus());
});

// Initialize recommendation service on server start
(async () => {
  try {
    console.log("🚀 Starting server with recommendation engine...");

    // Load data for initialization
    const orders = await loadJSON(path.join(__dirname, "data", "orders.json"));
    const products = await loadJSON(
      path.join(__dirname, "data", "products.json")
    );

    // Initialize in background (don't block server start)
    recService.initialize(orders, products).catch((err) => {
      console.error("Failed to initialize recommendation service:", err);
    });
  } catch (error) {
    console.error("Startup error:", error);
  }
})();
```

---

## 💻 BƯỚC 6: Cập nhật Frontend

### 6.1. Sửa file `js/recommended.js`

```javascript
/**
 * Recommendation Page Frontend
 * Fetch và hiển thị recommendations từ API
 */

import { getCurrentUser } from "./auth.js";

// API Base URL
const API_BASE =
  window.location.origin.includes("localhost") ||
  window.location.origin.includes("127.0.0.1")
    ? "http://localhost:3000"
    : window.location.origin;

/**
 * Fetch recommendations từ backend
 */
async function fetchRecommendations() {
  const user = getCurrentUser();

  if (!user || !user.id) {
    console.warn("No user logged in, redirecting to login");
    window.location.href = "/client/login.html";
    return null;
  }

  try {
    const response = await fetch(
      `${API_BASE}/api/recommendations?userId=${user.id}`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch recommendations:", error);
    return null;
  }
}

/**
 * Create product card HTML
 */
function createProductCard(product) {
  const imageUrl = product.image || "../images/placeholder.jpg";
  const price = new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
  }).format(product.price);

  const reason = product.recommendationReason || "";

  return `
    <div class="product-card" data-product-id="${product.id}">
      <div class="product-card__image">
        <img src="${imageUrl}" alt="${product.name}" loading="lazy">
        ${reason ? `<div class="product-card__badge">${reason}</div>` : ""}
      </div>
      <div class="product-card__content">
        <h3 class="product-card__name">${product.name}</h3>
        <p class="product-card__price">${price}</p>
        <button class="btn btn--primary btn--small add-to-cart-btn" 
                data-product-id="${product.id}">
          Thêm vào giỏ
        </button>
      </div>
    </div>
  `;
}

/**
 * Render product grid
 */
function renderGrid(containerId, products) {
  const container = document.getElementById(containerId);

  if (!container) {
    console.error(`Container ${containerId} not found`);
    return;
  }

  if (!products || products.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>Chưa có gợi ý nào. Hãy mua sắm để nhận gợi ý cá nhân hóa!</p>
      </div>
    `;
    return;
  }

  container.innerHTML = products.map((p) => createProductCard(p)).join("");

  // Attach event listeners for "Add to Cart" buttons
  container.querySelectorAll(".add-to-cart-btn").forEach((btn) => {
    btn.addEventListener("click", handleAddToCart);
  });
}

/**
 * Render user signals (top categories chips)
 */
function renderUserSignals(signals) {
  const signalEl = document.getElementById("recSignal");

  if (!signalEl) return;

  if (!signals.hasHistory) {
    signalEl.innerHTML = '<span class="chip chip--muted">Người dùng mới</span>';
    return;
  }

  const chips = signals.topCategories
    .map((cat) => `<span class="chip">${cat}</span>`)
    .join("");

  signalEl.innerHTML = chips;
}

/**
 * Update status message
 */
function updateStatus(message, type = "info") {
  const statusEl = document.getElementById("recStatus");
  if (!statusEl) return;

  statusEl.textContent = message;
  statusEl.className = `muted status-${type}`;
}

/**
 * Main render function
 */
async function renderRecommendations() {
  updateStatus("Đang phân tích đơn hàng...", "loading");

  try {
    const data = await fetchRecommendations();

    if (!data) {
      updateStatus("Không thể tải gợi ý. Vui lòng thử lại sau.", "error");
      return;
    }

    // Render user signals
    renderUserSignals(data.userSignals);

    // Render grids
    renderGrid("recPersonalGrid", data.personal);
    renderGrid("recSimilarGrid", data.similar);
    renderGrid("recTrendingGrid", data.trending);

    // Update status
    const statusMsg = data.userSignals.hasHistory
      ? `Phân tích ${data.userSignals.purchaseCount} đơn hàng • ${data.personal.length} gợi ý cá nhân`
      : "Bắt đầu mua sắm để nhận gợi ý cá nhân hóa!";

    updateStatus(statusMsg, "success");

    console.log("✅ Recommendations rendered:", data.metadata);
  } catch (error) {
    console.error("Render error:", error);
    updateStatus("Đã xảy ra lỗi khi tải gợi ý.", "error");
  }
}

/**
 * Handle Add to Cart
 */
function handleAddToCart(event) {
  const btn = event.target;
  const productId = btn.dataset.productId;

  // TODO: Implement actual add to cart logic
  console.log("Add to cart:", productId);

  // Visual feedback
  btn.textContent = "Đã thêm ✓";
  btn.disabled = true;

  setTimeout(() => {
    btn.textContent = "Thêm vào giỏ";
    btn.disabled = false;
  }, 2000);
}

/**
 * Refresh button handler
 */
function setupRefreshButton() {
  const refreshBtn = document.getElementById("recRefreshBtn");

  if (!refreshBtn) return;

  refreshBtn.addEventListener("click", async () => {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Đang tải...";

    await renderRecommendations();

    refreshBtn.disabled = false;
    refreshBtn.textContent = "Làm mới";
  });
}

/**
 * Initialize page
 */
function init() {
  console.log("🎯 Initializing recommendation page...");

  setupRefreshButton();
  renderRecommendations();
}

// Run on page load
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
```

---

## 🎨 BƯỚC 7: Thêm CSS cho Product Cards

### 7.1. Thêm vào `css/style.css`

```css
/* Recommendation Page Styles */

.chip-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.chip {
  display: inline-block;
  padding: 6px 14px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  text-transform: capitalize;
}

.chip--muted {
  background: #e0e0e0;
  color: #666;
}

/* Product Grid */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 24px;
  margin-top: 24px;
}

/* Product Card */
.product-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s, box-shadow 0.2s;
  cursor: pointer;
}

.product-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.product-card__image {
  position: relative;
  width: 100%;
  height: 200px;
  overflow: hidden;
  background: #f5f5f5;
}

.product-card__image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.product-card__badge {
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
}

.product-card__content {
  padding: 16px;
}

.product-card__name {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.product-card__price {
  font-size: 18px;
  font-weight: 700;
  color: #e74c3c;
  margin: 0 0 12px 0;
}

.add-to-cart-btn {
  width: 100%;
}

/* Empty State */
.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  padding: 60px 20px;
  color: #999;
}

.empty-state p {
  font-size: 16px;
  margin: 0;
}

/* Status Messages */
.status-loading {
  color: #3498db;
}

.status-success {
  color: #27ae60;
}

.status-error {
  color: #e74c3c;
}
```

---

## ✅ BƯỚC 8: Testing & Validation

### 8.1. Testing Workflow

#### **A. Backend Testing**

```bash
# 1. Start backend server
cd backoffice
node server.js

# Expected output:
# 🚀 Starting server with recommendation engine...
# 🔄 Training Apriori algorithm...
# 🔄 Training Matrix Factorization (SVD)...
# ✅ Apriori trained: X rules found
# ✅ Matrix Factorization trained: Y latent factors
# ✅ Recommendation Service initialized
# Server listening on port 3000
```

#### **B. API Testing**

1. **Test API endpoints:**

   ```bash
   # Get recommendations
   curl "http://localhost:3000/api/recommendations?userId=1"

   # Expected response structure:
   # {
   #   "personal": [...],      // Matrix Factorization results
   #   "similar": [...],       // Apriori results
   #   "trending": [...],      // Popularity-based
   #   "userSignals": {
   #     "topCategories": ["Rau củ", "Trái cây"],
   #     "purchaseCount": 5,
   #     "hasHistory": true
   #   },
   #   "metadata": { ... }
   # }

   # Check status
   curl "http://localhost:3000/api/recommendations/status"

   # Expected response:
   # {
   #   "isInitialized": true,
   #   "isTraining": false,
   #   "apriori": { "totalRules": 150 },
   #   "matrixFactorization": { "numUsers": 50, "numItems": 200 }
   # }

   # Manual retrain
   curl -X POST "http://localhost:3000/api/recommendations/retrain"
   ```

#### **C. Frontend Testing**

3. **Test trong browser:**
   - Mở `http://localhost:3000/html/recommended.html`
   - Đăng nhập với user có lịch sử mua hàng
   - Kiểm tra 3 sections hiển thị đúng
   - Click "Làm mới" để test refresh

### 8.2. Validation Checklist

#### **Backend Validation**

- [ ] Server starts without errors
- [ ] Apriori engine trains successfully (rules > 0)
- [ ] Matrix Factorization trains successfully (no SVD errors)
- [ ] Cache file được tạo: `backoffice/ml/services/recommendation/recommendation_cache.json`
- [ ] API endpoints trả về đúng format
- [ ] Cold start users (no history) nhận trending products
- [ ] Returning users nhận personalized + similar recommendations
- [ ] Response time < 1s cho API calls

#### **Frontend Validation**

- [ ] 3 sections render correctly (Personal, Similar, Trending)
- [ ] User signals (chips) hiển thị top categories
- [ ] Product cards hiển thị đầy đủ: image, name, price, reason
- [ ] "Thêm vào giỏ" button có visual feedback
- [ ] Loading state hiển thị khi fetching
- [ ] Error state hiển thị khi API fail
- [ ] "Làm mới" button hoạt động
- [ ] Responsive trên mobile (< 768px)

#### **Algorithm Validation**

- [ ] **Apriori**: Rules có lift > 1.2 (meaningful associations)
- [ ] **Matrix Factorization**: SVD converges (không có NaN values)
- [ ] **Hybrid**: Personal + Similar + Trending đều có kết quả
- [ ] **Filtering**: Không gợi ý sản phẩm đã mua

### 8.3. Performance Benchmarks

| Metric                | Target  | Acceptable |
| --------------------- | ------- | ---------- |
| API Response Time     | < 300ms | < 1000ms   |
| Initial Training Time | < 20s   | < 45s      |
| Frontend Load Time    | < 2s    | < 3s       |
| Memory Usage (Node)   | < 300MB | < 800MB    |
| Cache File Size       | < 3MB   | < 8MB      |

#### **Performance Testing**

```bash
# Test API response time
time curl "http://localhost:3000/api/recommendations?userId=1"

# Monitor memory usage (Windows PowerShell)
Get-Process -Name node | Select-Object WorkingSet

# Expected: ~150-400 MB
```

---

## 📊 BƯỚC 9: Monitoring & Optimization

### 9.1. Health Monitoring

#### **Real-time Dashboard**

Add to `backoffice/index.html`:

```javascript
// Monitor recommendation system health
async function monitorRecommendationHealth() {
  const status = await fetch("/api/recommendations/status").then((r) =>
    r.json()
  );

  console.table({
    "System Status": status.isInitialized ? "✅ Healthy" : "❌ Down",
    Training: status.isTraining ? "🔄 In Progress" : "✅ Ready",
    "Apriori Rules": status.apriori?.totalRules || 0,
    "MF Users": status.matrixFactorization?.numUsers || 0,
    "MF Items": status.matrixFactorization?.numItems || 0,
    "Last Updated": status.lastTrainTime
      ? new Date(status.lastTrainTime).toLocaleString()
      : "Never",
  });
}

// Run every 5 minutes
setInterval(monitorRecommendationHealth, 5 * 60 * 1000);
```

#### **Performance Logging**

```javascript
// Enhanced logging in recommendation_service.js
async getRecommendations(userId, maxRecommendations = 10) {
  const startTime = Date.now();

  try {
    // ... recommendation logic ...

    const executionTime = Date.now() - startTime;
    console.log({
      timestamp: new Date().toISOString(),
      userId: userId,
      personalCount: personalRecs.length,
      similarCount: similarRecs.length,
      trendingCount: trendingRecs.length,
      executionTime: executionTime,
      cacheHit: this.cache.recommendations[userId] ? true : false
    });

    // Alert if too slow
    if (executionTime > 1000) {
      console.warn(`⚠️ Slow recommendation response: ${executionTime}ms for user ${userId}`);
    }

  } catch (error) {
    console.error('❌ Recommendation error:', error);
    throw error;
  }
}
```

### 9.2. Performance Optimization

#### **A. Caching Strategy**

```javascript
// In recommendation_service.js
class RecommendationService {
  constructor() {
    this.cache = {
      recommendations: {},
      trending: null,
      trendingExpiry: null,
    };
    this.CACHE_TTL = 5 * 60 * 1000; // 5 minutes
  }

  async getRecommendations(userId, maxRecommendations = 10) {
    // Check user-specific cache
    const cached = this.cache.recommendations[userId];
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      console.log(`🎯 Cache HIT for user ${userId}`);
      return cached.data;
    }

    // Generate recommendations
    const recommendations = await this.generateRecommendations(
      userId,
      maxRecommendations
    );

    // Update cache
    this.cache.recommendations[userId] = {
      data: recommendations,
      timestamp: Date.now(),
    };

    // Limit cache size (keep last 1000 users)
    const cacheKeys = Object.keys(this.cache.recommendations);
    if (cacheKeys.length > 1000) {
      const oldestKey = cacheKeys[0];
      delete this.cache.recommendations[oldestKey];
    }

    return recommendations;
  }

  getTrendingProducts(count = 10) {
    // Cache trending products (changes less frequently)
    if (this.cache.trending && Date.now() < this.cache.trendingExpiry) {
      return this.cache.trending;
    }

    const trending = this.calculateTrendingProducts(count);
    this.cache.trending = trending;
    this.cache.trendingExpiry = Date.now() + 30 * 60 * 1000; // 30 minutes

    return trending;
  }
}
```

#### **B. Background Training**

```javascript
// In backoffice/server.js
class Server {
  async startServer() {
    // Initial training
    await this.recommendationService.initialize();

    // Schedule automatic retraining every 24 hours
    setInterval(async () => {
      const now = new Date();
      if (now.getHours() === 3) {
        // 3 AM
        console.log("🔄 Starting scheduled model retraining...");
        try {
          await this.recommendationService.retrain();
          console.log("✅ Scheduled retraining completed");
        } catch (error) {
          console.error("❌ Scheduled retraining failed:", error);
        }
      }
    }, 60 * 60 * 1000); // Check every hour
  }
}
```

#### **C. Incremental Updates**

```javascript
// Add to recommendation_service.js
async addNewOrder(order) {
  // Quick update without full retrain
  const userId = order.userId;

  // Invalidate user cache
  delete this.cache.recommendations[userId];

  // Update Apriori incrementally (if order has co-purchased items)
  if (order.items.length >= 2) {
    const transactions = this.prepareTransactions([order]);
    const newRules = this.aprioriEngine.mineRules(transactions);
    this.mergeRules(newRules); // Merge with existing rules
  }

  // For MF, queue for next retrain (too expensive to update incrementally)
  this.pendingOrders.push(order);

  // Trigger retrain if many pending orders
  if (this.pendingOrders.length >= 20) {
    console.log('🔄 Triggering incremental retrain (20 new orders)');
    await this.retrain();
  }
}
```

### 9.3. A/B Testing Framework

```javascript
// Add to backoffice/server.js
class ABTestingService {
  constructor() {
    this.variants = {
      A: { personal: 0.5, similar: 0.3, trending: 0.2 }, // Current
      B: { personal: 0.6, similar: 0.2, trending: 0.2 }, // More personal
      C: { personal: 0.4, similar: 0.4, trending: 0.2 }, // More similar
    };
    this.metrics = {};
  }

  assignVariant(userId) {
    // Assign variant based on user ID (consistent assignment)
    const hash = userId
      .toString()
      .split("")
      .reduce((a, b) => {
        a = (a << 5) - a + b.charCodeAt(0);
        return a & a;
      }, 0);

    const variantIndex = Math.abs(hash) % 3;
    return ["A", "B", "C"][variantIndex];
  }

  trackClick(userId, productId, variant, source) {
    const key = `${variant}_${source}`;
    if (!this.metrics[key]) {
      this.metrics[key] = { impressions: 0, clicks: 0 };
    }
    this.metrics[key].clicks++;
  }

  trackImpression(userId, productIds, variant) {
    const key = `${variant}_overall`;
    if (!this.metrics[key]) {
      this.metrics[key] = { impressions: 0, clicks: 0 };
    }
    this.metrics[key].impressions += productIds.length;
  }

  getResults() {
    const results = {};
    for (const [key, data] of Object.entries(this.metrics)) {
      results[key] = {
        ...data,
        ctr:
          data.impressions > 0
            ? ((data.clicks / data.impressions) * 100).toFixed(2) + "%"
            : "0%",
      };
    }
    return results;
  }
}

// Usage in recommendation endpoint
app.get("/api/recommendations", async (req, res) => {
  const userId = req.query.userId;
  const variant = abTesting.assignVariant(userId);
  const weights = abTesting.variants[variant];

  const recommendations = await recommendationService.getRecommendations(
    userId,
    10,
    weights
  );

  abTesting.trackImpression(
    userId,
    [
      ...recommendations.personal,
      ...recommendations.similar,
      ...recommendations.trending,
    ].map((p) => p.id),
    variant
  );

  res.json({ ...recommendations, variant }); // Include variant in response
});

// Track clicks
app.post("/api/recommendations/click", (req, res) => {
  const { userId, productId, source, variant } = req.body;
  abTesting.trackClick(userId, productId, variant, source);
  res.json({ success: true });
});

// View results
app.get("/api/recommendations/ab-results", (req, res) => {
  res.json(abTesting.getResults());
});
```

### 9.4. Key Metrics Dashboard

| Metric                   | Target  | Current | Status |
| ------------------------ | ------- | ------- | ------ |
| **Performance**          |
| API Response Time        | < 300ms | Monitor | ⏱️     |
| Cache Hit Rate           | > 80%   | Monitor | 📊     |
| Memory Usage             | < 300MB | Monitor | 💾     |
| **Quality**              |
| Apriori Rules            | 50-500  | Monitor | 📏     |
| MF Training Error        | < 0.1   | Monitor | 📉     |
| Recommendation Diversity | > 0.7   | Monitor | 🎲     |
| **Engagement**           |
| CTR (Personal)           | > 5%    | Monitor | 👆     |
| CTR (Similar)            | > 3%    | Monitor | 👆     |
| CTR (Trending)           | > 2%    | Monitor | 👆     |

---

## 🚀 BƯỚC 10: Production Deployment

### 10.1. Production Checklist

#### **Pre-Deployment**

- [ ] All tests passing (API, frontend, algorithms)
- [ ] Performance benchmarks met (< 300ms response time)
- [ ] Data quality validated (see 10.5)
- [ ] Cache strategy implemented
- [ ] Error handling robust
- [ ] Logging configured
- [ ] Environment variables setup
- [ ] Documentation complete

#### **Deployment Steps**

1. **Setup environment variables:**

```bash
# Create .env file in backoffice/
NODE_ENV=production
RECOMMENDATION_MIN_SUPPORT=0.02
RECOMMENDATION_MIN_CONFIDENCE=0.25
RECOMMENDATION_N_FACTORS=15
RECOMMENDATION_CACHE_TTL=300
RECOMMENDATION_MAX_ITERATIONS=50
RECOMMENDATION_LEARNING_RATE=0.01
```

2. **Configure production server:**

```javascript
// backoffice/server.js
const PORT = process.env.PORT || 3000;
const IS_PRODUCTION = process.env.NODE_ENV === "production";

if (IS_PRODUCTION) {
  // Enable production optimizations
  app.use(compression()); // Compress responses
  app.use(helmet()); // Security headers

  // Rate limiting
  const rateLimit = require("express-rate-limit");
  const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
  });
  app.use("/api/", limiter);

  // Increase cache TTL in production
  recommendationService.CACHE_TTL = 10 * 60 * 1000; // 10 minutes
}
```

3. **Install production dependencies:**

```bash
cd backoffice
npm install --production
npm install compression helmet express-rate-limit
```

### 10.2. Automated Retraining

#### **Windows Task Scheduler**

Create scheduled task for daily model retraining:

```powershell
# Run as Administrator in PowerShell

# Method 1: Using curl
$action = New-ScheduledTaskAction -Execute "curl.exe" -Argument "-X POST http://localhost:3000/api/recommendations/retrain"
$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3
Register-ScheduledTask -TaskName "RecommendationRetrain" -Action $action -Trigger $trigger -Settings $settings -Description "Daily ML model retraining at 3 AM"

# Method 2: Using PowerShell script
# Create retrain.ps1:
$response = Invoke-RestMethod -Uri "http://localhost:3000/api/recommendations/retrain" -Method Post
Write-Host "Retrain completed: $($response | ConvertTo-Json)"

# Then schedule it:
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\path\to\retrain.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 3:00AM
Register-ScheduledTask -TaskName "RecommendationRetrain" -Action $action -Trigger $trigger
```

#### **Node.js Cron (Alternative)**

```bash
npm install node-cron
```

```javascript
// Add to backoffice/server.js
const cron = require("node-cron");

// Schedule retraining every day at 3 AM
cron.schedule("0 3 * * *", async () => {
  console.log("🔄 Starting scheduled model retraining...");
  try {
    await recommendationService.retrain();
    console.log("✅ Scheduled retraining completed successfully");
  } catch (error) {
    console.error("❌ Scheduled retraining failed:", error);
    // TODO: Send alert email/notification
  }
});

console.log("⏰ Cron job registered: Daily retrain at 3:00 AM");
```

### 10.3. Monitoring & Alerting

#### **Health Check Endpoint**

```javascript
// Add to backoffice/server.js
app.get("/api/health", async (req, res) => {
  const health = {
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    recommendation: {
      initialized: recommendationService.isInitialized,
      lastTrain: recommendationService.lastTrainTime,
      cacheSize: Object.keys(recommendationService.cache.recommendations)
        .length,
    },
  };

  // Check if system is healthy
  if (!recommendationService.isInitialized) {
    health.status = "unhealthy";
    health.reason = "Recommendation service not initialized";
    return res.status(503).json(health);
  }

  // Check if memory usage is too high
  const memoryMB = health.memory.heapUsed / 1024 / 1024;
  if (memoryMB > 800) {
    health.status = "warning";
    health.reason = `High memory usage: ${memoryMB.toFixed(2)}MB`;
    return res.status(200).json(health);
  }

  res.json(health);
});
```

#### **Error Alerting (Email/Webhook)**

```javascript
// Add to backoffice/server.js
async function sendAlert(severity, message, details) {
  console.error(`🚨 ALERT [${severity}]: ${message}`, details);

  // TODO: Implement email/webhook notification
  // Example: Send to Slack, Discord, Email

  if (process.env.WEBHOOK_URL) {
    try {
      await fetch(process.env.WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          severity,
          message,
          details,
          timestamp: new Date().toISOString(),
        }),
      });
    } catch (error) {
      console.error("Failed to send alert:", error);
    }
  }
}

// Use in error handlers
app.post("/api/recommendations/retrain", async (req, res) => {
  try {
    await recommendationService.retrain();
    res.json({ success: true });
  } catch (error) {
    await sendAlert("critical", "Model retraining failed", {
      error: error.message,
    });
    res.status(500).json({ success: false, error: error.message });
  }
});
```

### 10.4. Backup & Recovery

#### **Automatic Cache Backup**

```javascript
// Add to recommendation_service.js
const fs = require("fs");
const path = require("path");

class RecommendationService {
  saveCache() {
    try {
      // Create backup directory
      const backupDir = path.join(__dirname, "recommendation", "backups");
      if (!fs.existsSync(backupDir)) {
        fs.mkdirSync(backupDir, { recursive: true });
      }

      // Save current cache with timestamp
      const timestamp = new Date()
        .toISOString()
        .replace(/:/g, "-")
        .split(".")[0];
      const backupPath = path.join(backupDir, `cache_${timestamp}.json`);
      fs.writeFileSync(backupPath, JSON.stringify(this.cache, null, 2));

      // Keep only last 7 backups
      const backups = fs
        .readdirSync(backupDir)
        .filter((f) => f.startsWith("cache_"))
        .sort()
        .reverse();

      if (backups.length > 7) {
        backups.slice(7).forEach((f) => {
          fs.unlinkSync(path.join(backupDir, f));
        });
      }

      console.log(`✅ Cache backed up: ${backupPath}`);
    } catch (error) {
      console.error("❌ Cache backup failed:", error);
    }
  }

  async retrain() {
    // Backup before retrain
    this.saveCache();

    try {
      // ... retraining logic ...
      console.log("✅ Retraining successful");
    } catch (error) {
      console.error("❌ Retraining failed, restoring from backup...");
      this.restoreFromBackup();
      throw error;
    }
  }

  restoreFromBackup() {
    try {
      const backupDir = path.join(__dirname, "recommendation", "backups");
      const backups = fs
        .readdirSync(backupDir)
        .filter((f) => f.startsWith("cache_"))
        .sort()
        .reverse();

      if (backups.length > 0) {
        const latestBackup = path.join(backupDir, backups[0]);
        this.cache = JSON.parse(fs.readFileSync(latestBackup, "utf8"));
        console.log(`✅ Restored from backup: ${backups[0]}`);
      }
    } catch (error) {
      console.error("❌ Backup restoration failed:", error);
    }
  }
}
```

### 10.5. Data Quality Validation

```javascript
// Add to backoffice/server.js
function validateTrainingData(orders, users, products) {
  const stats = {
    totalOrders: orders.length,
    uniqueUsers: new Set(orders.map((o) => o.userId)).size,
    uniqueProducts: new Set(
      orders.flatMap((o) => o.items.map((i) => i.productId))
    ).size,
    avgOrdersPerUser: orders.length / new Set(orders.map((o) => o.userId)).size,
    avgItemsPerOrder:
      orders.reduce((sum, o) => sum + o.items.length, 0) / orders.length,
  };

  const issues = [];
  const warnings = [];

  // Critical issues
  if (stats.totalOrders < 20)
    issues.push("Not enough orders (need 20+, recommended 100+)");
  if (stats.uniqueUsers < 10)
    issues.push("Not enough users (need 10+, recommended 50+)");
  if (stats.uniqueProducts < 30)
    issues.push("Not enough products (need 30+, recommended 200+)");

  // Warnings
  if (stats.avgOrdersPerUser < 2)
    warnings.push("Low user engagement (avg orders/user < 2)");
  if (stats.avgItemsPerOrder < 3)
    warnings.push("Small basket size (avg items/order < 3)");

  console.log("📊 Training Data Stats:", stats);

  if (issues.length > 0) {
    console.error("❌ Critical Data Quality Issues:", issues);
    sendAlert("critical", "Training data quality issues", { issues, stats });
    return false;
  }

  if (warnings.length > 0) {
    console.warn("⚠️ Data Quality Warnings:", warnings);
  } else {
    console.log("✅ Training data quality is good");
  }

  return true;
}

// Use before training
async function initialize() {
  const orders = await loadOrders();
  const users = await loadUsers();
  const products = await loadProducts();

  if (!validateTrainingData(orders, users, products)) {
    throw new Error("Training data quality check failed");
  }

  // Proceed with training...
}
```

### 10.6. Performance Tuning Guide

#### **Hyperparameter Tuning Table**

| Scenario             | minSupport | minConfidence | minLift | latentFactors | learningRate | regularization |
| -------------------- | ---------- | ------------- | ------- | ------------- | ------------ | -------------- |
| **Too Few Rules**    | ↓ 0.005    | ↓ 0.15        | -       | -             | -            | -              |
| **Too Many Rules**   | ↑ 0.03     | ↑ 0.35        | ↑ 1.5   | -             | -            | -              |
| **Poor MF Quality**  | -          | -             | -       | ↑ 20          | ↓ 0.005      | ↓ 0.01         |
| **Overfitting**      | -          | -             | -       | ↓ 10          | -            | ↑ 0.05         |
| **Slow Convergence** | -          | -             | -       | -             | ↑ 0.02       | -              |
| **Production**       | 0.02       | 0.25          | 1.2     | 15            | 0.01         | 0.02           |

#### **Memory Optimization**

```javascript
// Limit cache size in recommendation_service.js
const MAX_CACHE_SIZE = 1000; // users
const MAX_CACHE_AGE = 30 * 60 * 1000; // 30 minutes

pruneCache() {
  const now = Date.now();
  const entries = Object.entries(this.cache.recommendations);

  // Remove expired entries
  const valid = entries.filter(([_, data]) => now - data.timestamp < MAX_CACHE_AGE);

  // Keep only most recent if still too large
  if (valid.length > MAX_CACHE_SIZE) {
    valid.sort((a, b) => b[1].timestamp - a[1].timestamp);
    valid.splice(MAX_CACHE_SIZE);
  }

  this.cache.recommendations = Object.fromEntries(valid);
}
```

### 10.7. Production Monitoring Script

```powershell
# monitor.ps1 - Run continuously in production
while ($true) {
  Write-Host "`n=== Recommendation System Status ===" -ForegroundColor Cyan
  Write-Host "Time: $(Get-Date)" -ForegroundColor Gray

  # Check health
  try {
    $health = Invoke-RestMethod -Uri "http://localhost:3000/api/health"
    Write-Host "Status: $($health.status)" -ForegroundColor $(if ($health.status -eq "healthy") { "Green" } else { "Yellow" })
    Write-Host "Uptime: $([math]::Round($health.uptime / 3600, 2)) hours"
    Write-Host "Memory: $([math]::Round($health.memory.heapUsed / 1024 / 1024, 2)) MB"
    Write-Host "Cache Size: $($health.recommendation.cacheSize) users"
  } catch {
    Write-Host "Status: UNHEALTHY" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
  }

  # Check recommendation status
  try {
    $status = Invoke-RestMethod -Uri "http://localhost:3000/api/recommendations/status"
    Write-Host "Apriori Rules: $($status.apriori.totalRules)"
    Write-Host "MF Users: $($status.matrixFactorization.numUsers)"
    Write-Host "MF Items: $($status.matrixFactorization.numItems)"
  } catch {
    Write-Host "Could not fetch recommendation status" -ForegroundColor Red
  }

  Start-Sleep -Seconds 60
}
```

**Run in background:**

```powershell
Start-Process PowerShell -ArgumentList "-File monitor.ps1" -WindowStyle Minimized
```

---

## 📚 Tài liệu tham khảo

### Apriori Algorithm

- [Apriori Algorithm Explained](https://en.wikipedia.org/wiki/Apriori_algorithm)
- [Association Rule Mining](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html)

### Matrix Factorization

- [Matrix Factorization for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [SVD for Collaborative Filtering](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)

### Hybrid Systems

- [Hybrid Recommendation Systems](https://link.springer.com/article/10.1007/s10115-010-0293-1)

---

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **"Module not found: ml-matrix"**
   → Chạy: `npm install ml-matrix ml-pca apriori`

2. **"SVD failed"**
   → Matrix quá sparse, giảm `nFactors` hoặc thêm dummy data

3. **"No recommendations returned"**
   → Check user có orders không, check cache file có tồn tại không

4. **Frontend không fetch được data**
   → Check CORS settings trong server.js
   → Verify API_BASE URL đúng

5. **Model train quá lâu**
   → Giảm số orders test
   → Giảm `nFactors` và `iterations`

---

## ✨ Advanced Features (Phase 2 - Optional)

### 11.1. Diversity & Serendipity

**Problem:** Recommendations quá giống nhau, thiếu đa dạng

**Solution:** Thêm diversity score vào ranking

```javascript
// Add to recommendation_service.js
calculateDiversityScore(recommendations, userHistory) {
  const userCategories = new Set(userHistory.flatMap(p => p.categories));

  return recommendations.map(rec => {
    // Category diversity (50%)
    const newCategories = rec.categories.filter(c => !userCategories.has(c));
    const categoryDiversity = newCategories.length / rec.categories.length;

    // Price diversity (30%) - different from average purchase
    const avgPrice = userHistory.reduce((sum, p) => sum + p.price, 0) / userHistory.length;
    const priceDiff = Math.abs(rec.price - avgPrice) / avgPrice;
    const priceDiversity = Math.min(priceDiff, 1);

    // Brand diversity (20%)
    const userBrands = new Set(userHistory.map(p => p.brand));
    const brandDiversity = userBrands.has(rec.brand) ? 0 : 1;

    const diversityScore =
      categoryDiversity * 0.5 +
      priceDiversity * 0.3 +
      brandDiversity * 0.2;

    return {
      ...rec,
      diversityScore,
      finalScore: rec.score * 0.8 + diversityScore * 0.2 // Balance relevance + diversity
    };
  });
}

// Add exploration (10% random)
async getRecommendations(userId, maxRecommendations = 10) {
  const recommendations = await this.generateRecommendations(userId, maxRecommendations);

  // Replace 10% with random products (serendipity)
  const explorationCount = Math.ceil(maxRecommendations * 0.1);
  const randomProducts = this.getRandomProducts(explorationCount, recommendations);

  recommendations.personal = [
    ...recommendations.personal.slice(0, -explorationCount),
    ...randomProducts
  ];

  return recommendations;
}
```

### 11.2. Explainability

**Problem:** User không biết tại sao được gợi ý sản phẩm này

**Solution:** Thêm reason vào mỗi recommendation

```javascript
// Enhanced recommendation with explanations
generateExplanation(product, source, userId) {
  const userHistory = this.getUserPurchaseHistory(userId);

  switch (source) {
    case 'personal':
      // Matrix Factorization - similar taste
      const similarUsers = this.findSimilarUsers(userId, 5);
      return {
        reason: 'Dựa trên sở thích của bạn',
        details: `${similarUsers.length} người dùng có sở thích tương tự đã mua sản phẩm này`,
        confidence: 0.85
      };

    case 'similar':
      // Apriori - frequently bought together
      const coProducts = this.findCoOccurringProducts(product.id, userHistory);
      if (coProducts.length > 0) {
        return {
          reason: `Thường mua cùng với ${coProducts[0].name}`,
          details: `${(coProducts[0].confidence * 100).toFixed(0)}% khách hàng mua cả hai`,
          confidence: coProducts[0].confidence
        };
      }
      return {
        reason: 'Sản phẩm liên quan',
        details: 'Dựa trên lịch sử mua hàng của bạn',
        confidence: 0.65
      };

    case 'trending':
      const salesCount = this.getProductSalesCount(product.id, 7); // last 7 days
      return {
        reason: 'Đang thịnh hành',
        details: `${salesCount} người đã mua trong tuần qua`,
        confidence: 0.75
      };

    default:
      return {
        reason: 'Gợi ý cho bạn',
        details: '',
        confidence: 0.5
      };
  }
}

// Include in response
async getRecommendations(userId, maxRecommendations = 10) {
  // ... generate recommendations ...

  return {
    personal: personalRecs.map(p => ({
      ...p,
      explanation: this.generateExplanation(p, 'personal', userId)
    })),
    similar: similarRecs.map(p => ({
      ...p,
      explanation: this.generateExplanation(p, 'similar', userId)
    })),
    trending: trendingRecs.map(p => ({
      ...p,
      explanation: this.generateExplanation(p, 'trending', userId)
    }))
  };
}
```

**Frontend display:**

```javascript
// In js/recommended.js
function createProductCard(product, source) {
  return `
    <div class="product-card">
      <img src="${product.image}" alt="${product.name}">
      <h3>${product.name}</h3>
      <p class="price">${formatPrice(product.price)}</p>
      
      <!-- Explanation badge -->
      <div class="explanation" title="${product.explanation?.details}">
        <i class="icon-info"></i>
        <span>${product.explanation?.reason || "Gợi ý cho bạn"}</span>
        ${
          product.explanation?.confidence
            ? `<span class="confidence">${(
                product.explanation.confidence * 100
              ).toFixed(0)}%</span>`
            : ""
        }
      </div>
      
      <button onclick="addToCart(${product.id})">Thêm vào giỏ</button>
    </div>
  `;
}
```

```css
/* In assets/css/reviews.css */
.explanation {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  font-size: 12px;
  color: white;
  margin: 8px 0;
  cursor: help;
  transition: transform 0.2s;
}

.explanation:hover {
  transform: scale(1.05);
}

.explanation .confidence {
  margin-left: auto;
  font-weight: bold;
  opacity: 0.9;
}
```

### 11.3. Real-time Updates

**Problem:** Recommendations chỉ update khi refresh page

**Solution:** Socket.IO cho real-time push

```bash
npm install socket.io
```

```javascript
// In backoffice/server.js
const socketIo = require("socket.io");
const io = socketIo(server);

io.on("connection", (socket) => {
  console.log(`User connected: ${socket.id}`);

  socket.on("subscribe", (userId) => {
    socket.join(`user_${userId}`);
    console.log(`User ${userId} subscribed to recommendations`);
  });
});

// Emit when new order is placed
app.post("/api/orders", async (req, res) => {
  const order = req.body;

  // Save order...
  await saveOrder(order);

  // Update recommendations incrementally
  await recommendationService.addNewOrder(order);

  // Notify user
  const newRecommendations = await recommendationService.getRecommendations(
    order.userId
  );
  io.to(`user_${order.userId}`).emit(
    "recommendations_updated",
    newRecommendations
  );

  res.json({ success: true });
});
```

```javascript
// In js/recommended.js (frontend)
const socket = io("http://localhost:3000");

socket.on("connect", () => {
  const userId = getCurrentUserId();
  socket.emit("subscribe", userId);
  console.log("Subscribed to recommendation updates");
});

socket.on("recommendations_updated", (recommendations) => {
  console.log("📬 New recommendations received!");

  // Show notification
  showToast("Có gợi ý mới dành cho bạn! 🎉");

  // Update UI with new recommendations
  renderRecommendations(recommendations);
});

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.classList.add("show"), 100);
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
```

### 11.4. Context-Aware Recommendations

**Time-of-Day Awareness:**

```javascript
// Add to recommendation_service.js
getContextualBoost(product, context) {
  const hour = new Date().getHours();
  let boost = 1.0;

  // Morning (6-10 AM) - boost breakfast items
  if (hour >= 6 && hour < 10) {
    if (product.categories.includes('Đồ uống') && product.name.includes('Cà phê')) {
      boost = 1.3;
    }
    if (product.categories.includes('Bánh mì') || product.name.includes('Sữa')) {
      boost = 1.2;
    }
  }

  // Lunch (11 AM - 2 PM) - boost meal ingredients
  if (hour >= 11 && hour < 14) {
    if (product.categories.includes('Rau củ') || product.categories.includes('Thịt cá')) {
      boost = 1.25;
    }
  }

  // Evening (5-8 PM) - boost dinner items
  if (hour >= 17 && hour < 20) {
    if (product.categories.includes('Thịt cá') || product.categories.includes('Gia vị')) {
      boost = 1.3;
    }
  }

  // Weekend boost for snacks
  const isWeekend = [0, 6].includes(new Date().getDay());
  if (isWeekend && product.categories.includes('Snack')) {
    boost *= 1.15;
  }

  return boost;
}

// Apply in scoring
scoreRecommendations(recommendations) {
  return recommendations.map(rec => ({
    ...rec,
    contextBoost: this.getContextualBoost(rec, { time: Date.now() }),
    finalScore: rec.score * this.getContextualBoost(rec)
  })).sort((a, b) => b.finalScore - a.finalScore);
}
```

### 11.5. User Feedback Loop

**Collect implicit feedback:**

```javascript
// In backoffice/server.js
const userFeedback = {};

app.post("/api/recommendations/feedback", (req, res) => {
  const { userId, productId, action, source } = req.body;
  // action: 'click', 'add_to_cart', 'purchase', 'skip'

  if (!userFeedback[userId]) {
    userFeedback[userId] = [];
  }

  userFeedback[userId].push({
    productId,
    action,
    source, // 'personal', 'similar', 'trending'
    timestamp: Date.now(),
  });

  // Use feedback to adjust weights
  if (userFeedback[userId].length >= 10) {
    adjustWeightsBasedOnFeedback(userId);
  }

  res.json({ success: true });
});

function adjustWeightsBasedOnFeedback(userId) {
  const feedback = userFeedback[userId];

  // Calculate engagement per source
  const engagement = {
    personal: 0,
    similar: 0,
    trending: 0,
  };

  feedback.forEach((f) => {
    const weight =
      { click: 1, add_to_cart: 2, purchase: 3, skip: -1 }[f.action] || 0;
    engagement[f.source] += weight;
  });

  // Normalize to weights
  const total =
    Math.abs(engagement.personal) +
    Math.abs(engagement.similar) +
    Math.abs(engagement.trending);
  const weights = {
    personal: Math.max(0.2, engagement.personal / total),
    similar: Math.max(0.2, engagement.similar / total),
    trending: Math.max(0.1, engagement.trending / total),
  };

  // Normalize to sum = 1
  const sum = weights.personal + weights.similar + weights.trending;
  Object.keys(weights).forEach((k) => (weights[k] /= sum));

  console.log(`🎯 Adjusted weights for user ${userId}:`, weights);

  // Store personalized weights
  recommendationService.userWeights[userId] = weights;
}
```

### 11.6. Neural Collaborative Filtering (Future)

**Upgrade from Matrix Factorization to Deep Learning:**

```javascript
// Conceptual - requires TensorFlow.js
const tf = require("@tensorflow/tfjs-node");

class NeuralCollaborativeFiltering {
  constructor(numUsers, numItems, embeddingSize = 32) {
    // User embedding layer
    this.userEmbedding = tf.layers.embedding({
      inputDim: numUsers,
      outputDim: embeddingSize,
      name: "user_embedding",
    });

    // Item embedding layer
    this.itemEmbedding = tf.layers.embedding({
      inputDim: numItems,
      outputDim: embeddingSize,
      name: "item_embedding",
    });

    // Neural network layers
    this.model = tf.sequential({
      layers: [
        tf.layers.concatenate(),
        tf.layers.dense({ units: 128, activation: "relu" }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: "relu" }),
        tf.layers.dense({ units: 1, activation: "sigmoid" }),
      ],
    });

    this.model.compile({
      optimizer: "adam",
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
  }

  async train(interactions, epochs = 10) {
    // interactions: [{ userId, itemId, rating }]

    const userIds = tf.tensor1d(
      interactions.map((i) => i.userId),
      "int32"
    );
    const itemIds = tf.tensor1d(
      interactions.map((i) => i.itemId),
      "int32"
    );
    const ratings = tf.tensor1d(interactions.map((i) => i.rating));

    await this.model.fit([userIds, itemIds], ratings, {
      epochs,
      batchSize: 128,
      validationSplit: 0.2,
    });
  }

  predict(userId, itemIds) {
    const userTensor = tf.fill([itemIds.length], userId, "int32");
    const itemTensor = tf.tensor1d(itemIds, "int32");

    const predictions = this.model.predict([userTensor, itemTensor]);
    return predictions.arraySync();
  }
}
```

---

## 🎯 KẾT LUẬN

### ✅ Checklist Hoàn Thành

Sau khi hoàn thành tài liệu này, bạn sẽ có một hệ thống recommendation hoàn chỉnh:

#### **Core Components**

- [x] **Apriori Engine** - Association rule mining từ transaction history
  - Min Support: 0.02, Min Confidence: 0.25, Min Lift: 1.2
  - Tìm patterns "Frequently Bought Together"
  - ~150-500 rules tùy data size
- [x] **Matrix Factorization Engine** - Collaborative filtering với SVD
  - 15 latent factors, Learning rate: 0.01, Regularization: 0.02
  - Dự đoán user preferences dựa trên similar users
  - Handles sparse matrices efficiently
- [x] **Content-Based Filtering** - Category & feature similarity
  - Cosine similarity trên product embeddings
  - Bổ sung cho cold start items
- [x] **Hybrid Recommendation Service** - Kết hợp 3 algorithms
  - Weights: Personal (50%), Similar (30%), Trending (20%)
  - Adaptive weights dựa trên user engagement
  - Cache-enabled cho performance

#### **Backend Infrastructure**

- [x] **REST API** - 3 main endpoints + health check
  - `GET /api/recommendations?userId=X` - Get recommendations
  - `POST /api/recommendations/retrain` - Manual retrain
  - `GET /api/recommendations/status` - System status
  - `GET /api/health` - Health monitoring
- [x] **Caching System** - Multi-level caching
  - User recommendations cache (5-10 min TTL)
  - Trending products cache (30 min TTL)
  - Model cache (persistent between restarts)
  - Automatic pruning (max 1000 users)
- [x] **Automated Retraining** - Daily model updates
  - Scheduled at 3 AM via Task Scheduler / Cron
  - Incremental updates for new orders
  - Backup before retrain + rollback on failure

#### **Frontend Integration**

- [x] **Responsive UI** - 3-section layout
  - Personal Recommendations (Matrix Factorization)
  - Similar Products (Apriori)
  - Trending Products (Popularity)
- [x] **User Signals** - Visual feedback
  - Top categories chips
  - Purchase count badge
  - Loading & error states
- [x] **Interactive Features**
  - "Làm mới" button for manual refresh
  - "Thêm vào giỏ" with animation
  - Explanation tooltips (optional)

#### **Quality Assurance**

- [x] **Testing Workflow** - Comprehensive validation
  - Backend: API tests, algorithm validation, performance tests
  - Frontend: UI tests, responsive tests, interaction tests
  - Integration: End-to-end user journey tests
- [x] **Performance Benchmarks** - Production-ready metrics
  - API Response: < 300ms (target), < 1000ms (acceptable)
  - Training Time: < 20s (target), < 45s (acceptable)
  - Memory Usage: < 300MB (target), < 800MB (acceptable)
  - Cache Hit Rate: > 80%
- [x] **Monitoring & Alerting** - Real-time health checks
  - Automated health checks every 5 minutes
  - Email/webhook alerts for critical issues
  - Performance dashboard in backoffice

#### **Production Deployment**

- [x] **Environment Setup** - Production-ready config
  - Environment variables for all settings
  - Rate limiting (100 req/15min per IP)
  - Compression & security headers
  - HTTPS ready
- [x] **Backup & Recovery** - Data protection
  - Automatic cache backups (keep last 7)
  - Rollback on failed retrain
  - Data quality validation before training
- [x] **Maintenance Plan** - Long-term sustainability
  - Daily: Automated retraining
  - Weekly: Cache cleanup
  - Monthly: Performance review & hyperparameter tuning
  - Quarterly: Algorithm evaluation & upgrade planning

### 📊 System Metrics Summary

| Component                | Metric          | Target  |
| ------------------------ | --------------- | ------- |
| **Apriori**              | Rules Generated | 50-500  |
| **Apriori**              | Avg Lift        | > 1.2   |
| **Matrix Factorization** | Training Error  | < 0.1   |
| **Matrix Factorization** | Latent Factors  | 15      |
| **API**                  | Response Time   | < 300ms |
| **API**                  | Availability    | > 99%   |
| **Cache**                | Hit Rate        | > 80%   |
| **Memory**               | Heap Usage      | < 300MB |
| **Engagement**           | CTR (Personal)  | > 5%    |
| **Engagement**           | CTR (Similar)   | > 3%    |

### 🚀 Next Steps

#### **Immediate (Next 7 Days)**

1. ✅ Complete all 10 implementation steps
2. ✅ Run comprehensive testing (Step 8)
3. ✅ Deploy to production (Step 10)
4. ⏳ Monitor performance for 1 week
5. ⏳ Collect initial user feedback

#### **Short-term (1-3 Months)**

1. ⏳ Tune hyperparameters based on production data
2. ⏳ Implement explainability features (Step 11.2)
3. ⏳ Add diversity scoring (Step 11.1)
4. ⏳ Setup A/B testing (Step 9.3)
5. ⏳ Optimize for mobile users

#### **Long-term (3-12 Months)**

1. 📅 Real-time recommendations via Socket.IO (Step 11.3)
2. 📅 Context-aware recommendations (time, weather) (Step 11.4)
3. 📅 User-user collaborative filtering (currently only item-item)
4. 📅 Neural Collaborative Filtering upgrade (Step 11.6)
5. 📅 Multi-armed bandit for exploration/exploitation

### 🔗 Integration với Season Page

Hệ thống này (Recommended Page) có thể hoạt động song song với [SEASONAL_RECOMMENDATION_IMPLEMENTATION.md](SEASONAL_RECOMMENDATION_IMPLEMENTATION.md):

| Aspect         | Recommended Page       | Season Page                        |
| -------------- | ---------------------- | ---------------------------------- |
| **Algorithms** | CF + Apriori + Content | S-CF + S-Apriori + Trend + Weather |
| **Weights**    | 50% + 30% + 20%        | 30% + 30% + 25% + 15%              |
| **Context**    | User history           | Season + Time + Weather            |
| **Retraining** | Daily                  | Weekly                             |
| **Use Case**   | General shopping       | Seasonal shopping                  |

**Shared Components:**

- Base Matrix Factorization logic
- Base Apriori implementation
- Product embeddings
- Cache infrastructure
- API structure

### 📚 Learning Resources

#### **Algorithms**

- [Apriori Algorithm Tutorial](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html)
- [Matrix Factorization for RecSys (Netflix Paper)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [SVD for Collaborative Filtering (Google)](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
- [Hybrid Recommendation Systems (Springer)](https://link.springer.com/article/10.1007/s10115-010-0293-1)

#### **Implementation**

- [ml-matrix Documentation](https://github.com/mljs/matrix)
- [node-apriori Documentation](https://www.npmjs.com/package/apriori)
- [Express.js Best Practices](https://expressjs.com/en/advanced/best-practice-performance.html)
- [Node.js Performance Tips](https://nodejs.org/en/docs/guides/simple-profiling/)

#### **Production**

- [Node.js Production Checklist](https://github.com/i0natan/nodebestpractices)
- [API Rate Limiting Guide](https://www.npmjs.com/package/express-rate-limit)
- [PM2 Process Manager](https://pm2.keymetrics.io/)
- [Winston Logger](https://github.com/winstonjs/winston)

---

## 🐛 Common Issues & Solutions

### Issue: "Module not found: ml-matrix"

**Solution:**

```bash
cd backoffice
npm install ml-matrix ml-pca apriori mathjs
```

### Issue: "SVD failed to converge"

**Causes:** Matrix too sparse, NaN values, or insufficient data

**Solutions:**

1. Check data quality: `validateTrainingData()` (Step 10.5)
2. Reduce latent factors: `nFactors = 10`
3. Increase regularization: `lambda = 0.05`
4. Add dummy/synthetic data for cold start items

### Issue: "No recommendations returned"

**Diagnosis:**

```bash
# Check user has orders
curl "http://localhost:3000/api/recommendations/status"

# Check cache file exists
ls backoffice/ml/services/recommendation/recommendation_cache.json

# Check API response
curl "http://localhost:3000/api/recommendations?userId=1"
```

**Solutions:**

1. Verify user has purchase history (>= 1 order)
2. Check cache file is not corrupted
3. Manual retrain: `curl -X POST http://localhost:3000/api/recommendations/retrain`
4. Clear cache and restart server

### Issue: "Frontend can't fetch data (CORS)"

**Solution:**

```javascript
// In backoffice/server.js
const cors = require("cors");
app.use(
  cors({
    origin: ["http://localhost:3000", "http://127.0.0.1:3000"],
    credentials: true,
  })
);
```

### Issue: "Model training too slow (> 60s)"

**Solutions:**

1. Reduce dataset size for testing
2. Decrease `maxIterations` from 50 to 30
3. Decrease `nFactors` from 15 to 10
4. Use smaller `minSupport` sample (0.03 instead of 0.01)

### Issue: "Memory leak / increasing memory usage"

**Solutions:**

1. Implement cache pruning (see Step 10.6)
2. Restart server daily via Task Scheduler
3. Use `--max-old-space-size=512` flag:
   ```bash
   node --max-old-space-size=512 server.js
   ```
4. Profile with Chrome DevTools:
   ```bash
   node --inspect server.js
   # Open chrome://inspect
   ```

---

## 🎓 FAQ

**Q: Tại sao dùng 3 algorithms thay vì 1?**
A: Mỗi algorithm có strengths/weaknesses riêng:

- **Matrix Factorization:** Tốt cho personalization nhưng cần data
- **Apriori:** Tốt cho "Frequently Bought Together" nhưng chỉ dựa vào patterns
- **Content-Based:** Tốt cho cold start nhưng thiếu serendipity
  Hybrid system kết hợp strengths của cả 3.

**Q: Bao nhiêu data cần để train model?**
A: **Minimum:** 20 orders, 10 users, 30 products
**Recommended:** 100+ orders, 50+ users, 200+ products
**Optimal:** 500+ orders, 200+ users, 500+ products

**Q: Model cần retrain bao lâu một lần?**
A:

- **Daily (recommended):** Cho real-time updates
- **Weekly:** Nếu ít orders mới (< 10/ngày)
- **On-demand:** Khi có major data changes (new products, promotions)

**Q: Làm sao biết recommendations có tốt không?**
A: Track 3 metrics:

1. **CTR (Click-Through Rate):** % users click on recommendations (target: > 3%)
2. **Conversion Rate:** % users mua sản phẩm được gợi ý (target: > 1%)
3. **Diversity:** % recommendations từ categories khác nhau (target: > 0.7)

**Q: Cold start problem - user mới hoặc product mới?**
A:

- **User mới:** Show trending + popular products + content-based
- **Product mới:** Content-based filtering dựa trên category/features
- **Both:** Use demographic-based recommendations (age, location)

**Q: Có thể dùng cho millions of products không?**
A: Current implementation tốt cho:

- **Products:** < 10,000 (tốt), < 50,000 (acceptable)
- **Users:** < 100,000 (tốt), < 500,000 (acceptable)
- **Orders:** < 1,000,000

Nếu scale lớn hơn, cần:

- Use database (PostgreSQL/MongoDB) thay vì JSON
- Implement sharding cho Matrix Factorization
- Use Redis cho caching
- Consider Apache Spark cho distributed training

---

## 🏆 Success Criteria

Hệ thống được coi là **successful** khi:

✅ **Technical Metrics**

- [ ] API response time < 300ms (p95)
- [ ] System uptime > 99%
- [ ] Cache hit rate > 80%
- [ ] Memory usage stable (< 300MB)
- [ ] Zero critical errors in 7 days

✅ **Business Metrics**

- [ ] CTR > 3% across all recommendation types
- [ ] Conversion rate > 1% on recommended products
- [ ] Average basket size increase > 10% (vs non-recommended)
- [ ] User engagement time increase > 15%

✅ **Quality Metrics**

- [ ] Recommendation diversity > 0.7
- [ ] User satisfaction score > 4/5
- [ ] % relevant recommendations > 80% (user survey)
- [ ] Cold start coverage > 90% (all new users get recommendations)

---

**🎉 Chúc mừng! Bạn đã hoàn thành hệ thống Recommendation với Apriori & Matrix Factorization!**

_Tài liệu này được cập nhật: 23/12/2025_  
_Version: 2.0 (Enhanced)_  
_Tác giả: GitHub Copilot_  
_Review: Đã đồng bộ với SEASONAL_RECOMMENDATION_IMPLEMENTATION.md_

---
