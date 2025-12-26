import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class TimeAwareRecommender {
  constructor(options = {}) {
    const { dbPath, lambda = 0.05, lookbackDays = 90 } = options;
    this.dbPath = dbPath || path.join(__dirname, "../../db.json");
    this.lambda = lambda;
    this.lookbackDays = lookbackDays;
    this.msPerDay = 1000 * 60 * 60 * 24;
  }

  async loadData() {
    try {
      const raw = JSON.parse(await fs.readFile(this.dbPath, "utf-8"));
      const products = Array.isArray(raw.products) ? raw.products : [];
      const reviews = Array.isArray(raw.reviews) ? raw.reviews : [];
      return { products, reviews };
    } catch (err) {
      console.error(`TimeAwareRecommender loadData failed: ${err.message}`);
      return { products: [], reviews: [] };
    }
  }

  calculateDecayedRatings(reviews) {
    const now = Date.now();

    return reviews
      .map((review) => {
        const timestampMs = Date.parse(review.timestamp);
        if (Number.isNaN(timestampMs)) return null;

        const diffDays = Math.max(0, (now - timestampMs) / this.msPerDay);
        if (diffDays > this.lookbackDays) return null;

        const rating = Number(review.rating);
        if (!Number.isFinite(rating)) return null;

        const effectiveRating = rating * Math.exp(-this.lambda * diffDays);

        return {
          ...review,
          effectiveRating,
          diffDays,
        };
      })
      .filter(Boolean);
  }

  buildItemVectors(decayedReviews) {
    const vectors = new Map();

    decayedReviews.forEach((r) => {
      const productId = String(r.productId);
      const userId = String(r.userId);
      if (!vectors.has(productId)) {
        vectors.set(productId, {});
      }
      vectors.get(productId)[userId] = r.effectiveRating;
    });

    return vectors;
  }

  cosineSimilarity(vecA = {}, vecB = {}) {
    const users = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
    let dot = 0;
    let normA = 0;
    let normB = 0;

    users.forEach((userId) => {
      const a = vecA[userId] || 0;
      const b = vecB[userId] || 0;
      dot += a * b;
      normA += a * a;
      normB += b * b;
    });

    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  getPopularProducts(products, excludeIds = new Set(), limit = 5) {
    return [...products]
      .filter((p) => !excludeIds.has(String(p.id)))
      .sort((a, b) => (Number(b.popular) || 0) - (Number(a.popular) || 0))
      .slice(0, limit);
  }

  async getRecommendations(userId, topN = 5) {
    const { products, reviews } = await this.loadData();
    const decayed = this.calculateDecayedRatings(reviews);

    const ratedByUser = decayed.filter(
      (r) => String(r.userId) === String(userId)
    );
    const ratedProductIds = new Set(
      ratedByUser.map((r) => String(r.productId))
    );

    const productMap = new Map(products.map((p) => [String(p.id), p]));

    if (ratedByUser.length === 0 || decayed.length === 0) {
      return this.getPopularProducts(products, ratedProductIds, topN);
    }

    const itemVectors = this.buildItemVectors(decayed);
    const scored = [];

    for (const [candidateId, candidateVec] of itemVectors.entries()) {
      if (ratedProductIds.has(candidateId)) continue;

      let numerator = 0;
      let denominator = 0;

      ratedByUser.forEach((r) => {
        const ratedVec = itemVectors.get(String(r.productId)) || {};
        const sim = this.cosineSimilarity(ratedVec, candidateVec);
        if (sim <= 0) return;
        numerator += sim * r.effectiveRating;
        denominator += Math.abs(sim);
      });

      if (denominator > 0) {
        scored.push({
          productId: candidateId,
          score: numerator / denominator,
        });
      }
    }

    scored.sort((a, b) => b.score - a.score);

    const recommendations = [];
    const used = new Set(ratedProductIds);

    for (const item of scored) {
      if (recommendations.length >= topN) break;
      const product = productMap.get(String(item.productId));
      if (product && !used.has(String(product.id))) {
        recommendations.push(product);
        used.add(String(product.id));
      }
    }

    if (recommendations.length < topN) {
      const fillers = this.getPopularProducts(
        products,
        used,
        topN - recommendations.length
      );
      recommendations.push(...fillers);
    }

    return recommendations.slice(0, topN);
  }
}

export default TimeAwareRecommender;
