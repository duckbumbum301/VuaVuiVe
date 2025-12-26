/**
 * Matrix Factorization Engine
 * Collaborative Filtering sử dụng SVD/ALS
 */

import { Matrix, SingularValueDecomposition } from "ml-matrix";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

export default MatrixFactorizationEngine;
