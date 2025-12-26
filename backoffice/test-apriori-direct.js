/**
 * Direct test of Apriori with new thresholds
 */
import AprioriEngine from "./recommendation/apriori_engine.js";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function test() {
  console.log("🧪 Testing Apriori Engine directly...\n");

  // Load orders
  const ordersPath = path.join(__dirname, "data", "orders-ml.json");
  const orders = JSON.parse(await fs.readFile(ordersPath, "utf-8"));
  const completed = orders.filter((o) => o.status === "completed");

  // Create engine with lower thresholds
  console.log("📝 Creating AprioriEngine with config:");
  const engine = new AprioriEngine({
    minSupport: 0.02,
    minConfidence: 0.15,
    minLift: 1.0,
  });
  console.log(`   minSupport: ${engine.minSupport}`);
  console.log(`   minConfidence: ${engine.minConfidence}`);
  console.log(`   minLift: ${engine.minLift}\n`);

  // Train
  console.log(`🎓 Training with ${completed.length} orders...`);
  await engine.train(completed);

  // Show results
  console.log(`\n✅ Training completed!`);
  console.log(`📊 Total rules found: ${engine.rules.length}\n`);

  if (engine.rules.length > 0) {
    console.log("🏆 Top 10 Rules:");
    engine.rules.slice(0, 10).forEach((rule, i) => {
      console.log(
        `${i + 1}. [${rule.antecedent.join(",")}] → [${rule.consequent.join(
          ","
        )}]`
      );
      console.log(
        `   Confidence: ${(rule.confidence * 100).toFixed(
          1
        )}%, Lift: ${rule.lift.toFixed(2)}, Score: ${rule.score.toFixed(3)}`
      );
    });
  } else {
    console.log("❌ No rules found!");
    console.log("💡 Try lowering minLift further (e.g., 0.5)");
  }
}

test().catch(console.error);
