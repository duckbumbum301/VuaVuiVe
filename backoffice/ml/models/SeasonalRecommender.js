import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class SeasonalRecommender {
  constructor(options = {}) {
    const { dbPath, minSupport = 0.2, minConfidence = 0.5 } = options;
    this.dbPath = dbPath || path.join(__dirname, "../../db.json");
    this.minSupport = minSupport;
    this.minConfidence = minConfidence;
  }

  async loadData() {
    try {
      const raw = JSON.parse(await fs.readFile(this.dbPath, "utf-8"));
      const products = Array.isArray(raw.products) ? raw.products : [];
      const orders = Array.isArray(raw.orders) ? raw.orders : [];
      return { products, orders };
    } catch (err) {
      console.error(`SeasonalRecommender loadData failed: ${err.message}`);
      return { products: [], orders: [] };
    }
  }

  getSeason(date = new Date()) {
    const month = date.getMonth() + 1; // 1-12
    if (month >= 1 && month <= 3) return "Spring";
    if (month >= 4 && month <= 6) return "Summer";
    if (month >= 7 && month <= 9) return "Autumn";
    return "Winter";
  }

  filterOrdersBySeason(orders, season) {
    return orders.filter((order) => {
      const ts = order.createdAt || order.created_at || order.created_at;
      if (!ts) return false;
      const d = new Date(ts);
      if (Number.isNaN(d.getTime())) return false;
      return this.getSeason(d) === season;
    });
  }

  buildTransactions(seasonalOrders) {
    const transactions = [];

    seasonalOrders.forEach((order) => {
      const items = order.items;
      if (!items) return;

      const txSet = new Set();

      if (Array.isArray(items)) {
        items.forEach((item) => {
          const pid = item?.productId || item?.id || item?.product_id;
          if (pid !== undefined && pid !== null) {
            txSet.add(String(pid));
          }
        });
      } else if (typeof items === "object") {
        Object.keys(items).forEach((pid) => txSet.add(String(pid)));
      }

      if (txSet.size > 0) {
        transactions.push(Array.from(txSet));
      }
    });

    return transactions;
  }

  minePairRules(transactions, products) {
    if (!transactions.length)
      return { rules: [], supportCounts: {}, singleCounts: {} };

    const singleCounts = {};
    const pairCounts = {};
    const totalTx = transactions.length;

    transactions.forEach((tx) => {
      const unique = Array.from(new Set(tx));
      // Count singles
      unique.forEach((pid) => {
        singleCounts[pid] = (singleCounts[pid] || 0) + 1;
      });

      // Count pairs (unordered)
      for (let i = 0; i < unique.length; i += 1) {
        for (let j = i + 1; j < unique.length; j += 1) {
          const a = unique[i];
          const b = unique[j];
          const key = a < b ? `${a}__${b}` : `${b}__${a}`;
          pairCounts[key] = (pairCounts[key] || 0) + 1;
        }
      }
    });

    const productMap = new Map(products.map((p) => [String(p.id), p]));
    const rules = [];

    Object.entries(pairCounts).forEach(([key, countAB]) => {
      const [a, b] = key.split("__");
      const support = countAB / totalTx;
      if (support < this.minSupport) return;

      const countA = singleCounts[a] || 0;
      const countB = singleCounts[b] || 0;

      const confAB = countA ? countAB / countA : 0;
      const confBA = countB ? countAB / countB : 0;

      if (confAB >= this.minConfidence) {
        rules.push({
          triggerId: a,
          trigger: productMap.get(a)?.name || a,
          recommendId: b,
          recommend: productMap.get(b)?.name || b,
          confidence: Number(confAB.toFixed(3)),
          support: Number(support.toFixed(3)),
        });
      }

      if (confBA >= this.minConfidence) {
        rules.push({
          triggerId: b,
          trigger: productMap.get(b)?.name || b,
          recommendId: a,
          recommend: productMap.get(a)?.name || a,
          confidence: Number(confBA.toFixed(3)),
          support: Number(support.toFixed(3)),
        });
      }
    });

    rules.sort((x, y) => y.confidence - x.confidence || y.support - x.support);

    return { rules, supportCounts: pairCounts, singleCounts };
  }

  getPopularFallback(products, limit = 5) {
    const sorted = [...products].sort(
      (a, b) => (Number(b.popular) || 0) - (Number(a.popular) || 0)
    );

    const pairs = [];
    for (let i = 0; i < sorted.length && pairs.length < limit; i += 2) {
      const a = sorted[i];
      const b = sorted[i + 1];
      if (a && b) {
        pairs.push({
          triggerId: a.id,
          trigger: a.name,
          recommendId: b.id,
          recommend: b.name,
          confidence: 0,
          support: 0,
          fallback: true,
        });
      } else if (a) {
        pairs.push({
          triggerId: a.id,
          trigger: "Popular pick",
          recommendId: a.id,
          recommend: a.name,
          confidence: 0,
          support: 0,
          fallback: true,
        });
      }
    }

    return pairs.slice(0, limit);
  }

  async getSeasonalRules() {
    const now = new Date();
    const season = this.getSeason(now);
    const { products, orders } = await this.loadData();

    const seasonalOrders = this.filterOrdersBySeason(orders, season);
    const transactions = this.buildTransactions(seasonalOrders);

    const { rules } = this.minePairRules(transactions, products);

    if (!rules.length) {
      return { season, rules: this.getPopularFallback(products) };
    }

    return { season, rules };
  }
}

export default SeasonalRecommender;
