"""
Flask API cho Recommendation System
Expose endpoints để frontend gọi
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*unpickle.*')

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
from datetime import datetime, timezone
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from recommender import HybridRecommender
from vvv_adapter import VVVInstacartAdapter

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS cho frontend gọi được

# Load recommender (chỉ load 1 lần khi start server)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "03_features"

print("🔥 Loading recommendation model...")
recommender = HybridRecommender(MODELS_DIR, FEATURES_DIR)
print("✓ Model loaded successfully!")

# Load VVV Adapter
VVV_DATA_DIR = PROJECT_ROOT.parent / 'backoffice' / 'data'
MAPPING_FILE = PROJECT_ROOT / 'mappings' / 'vvv_instacart_mapping.json'

print("🔄 Loading VVV-Instacart adapter...")
adapter = VVVInstacartAdapter(VVV_DATA_DIR, MAPPING_FILE)
print("✓ Adapter loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Recommendation API is running'})


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Gợi ý sản phẩm cho VVV user
    
    Request body:
    {
        "user_id": 4,  // VVV user ID
        "n": 10,
        "filter_purchased": true
    }
    
    Response:
    {
        "user_id": 4,
        "recommendations": [
            {
                "product_id": 110,
                "score": 28.5,
                "name": "Cà rốt (500g)",
                "price": 15000,
                "image": "../images/VEG/root/Cà Rốt.jpg",
                "category": "veg/root",
                "reason": "Khách hàng thường mua cùng"
            },
            ...
        ]
    }
    """
    try:
        # Pick up newly synced backoffice/data/*.json changes (orders/products/users)
        # without needing to restart the Flask server.
        try:
            adapter.reload_if_changed()
        except Exception:
            pass

        data = request.get_json() or {}
        
        # Validate input
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id is required'}), 400

        # Frontend user_id (localStorage) may not match backoffice user ids.
        # Accept extra identity hints to resolve purchase history.
        vvv_user_id = data.get('user_id')
        try:
            vvv_user_id_int = int(vvv_user_id) if vvv_user_id is not None else None
        except Exception:
            vvv_user_id_int = None

        user_email = data.get('user_email')
        user_name = data.get('user_name')
        user_phone = data.get('user_phone')

        n = int(data.get('n', 10))
        filter_purchased = bool(data.get('filter_purchased', True))

        # Allow frontend to tune weights
        # Defaults are tuned for VVV proxy usage (we mainly trust basket co-occurrence)
        w_cf = float(data.get('w_cf', 0.0))
        w_basket = float(data.get('w_basket', 0.85))
        w_pop = float(data.get('w_pop', 0.15))

        # 1. Get purchase history using best available identity
        recent_orders = adapter.get_vvv_orders_by_identity(
            vvv_user_id=vvv_user_id_int,
            email=user_email,
            name=user_name,
            phone=user_phone,
            limit=80,
        )

        vvv_purchase_history = adapter.get_vvv_purchase_history_by_identity(
            vvv_user_id=vvv_user_id_int,
            email=user_email,
            name=user_name,
            phone=user_phone,
        )

        # Build a time-decayed preference profile from recent orders.
        # Recent orders should influence more than old ones; larger quantities also matter.
        def _parse_dt(raw: object):
            if not raw:
                return None
            s = str(raw).strip()
            if not s:
                return None
            try:
                if s.endswith('Z'):
                    s = s[:-1] + '+00:00'
                dt = datetime.fromisoformat(s)
            except Exception:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        decay_days = float(data.get('recency_decay_days', 21.0))
        decay_days = max(3.0, min(decay_days, 180.0))

        cat_counts = {}
        root_counts = {}
        for order in (recent_orders or []):
            dt = _parse_dt(order.get('createdAt') or order.get('created_at'))
            if dt:
                days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
            else:
                days_ago = 0.0
            w_time = math.exp(-days_ago / decay_days)

            for item in order.get('items', []) or []:
                pid = item.get('productId')
                if not pid:
                    continue
                try:
                    qty = int(item.get('quantity', 1))
                except Exception:
                    qty = 1
                qty = max(1, min(qty, 50))

                p = adapter.product_index.get(str(pid))
                if not p:
                    continue
                cat_key = adapter._get_product_cat_key(p)
                root = cat_key.split('/')[0] if '/' in cat_key else cat_key

                w = float(qty) * float(w_time)
                cat_counts[cat_key] = cat_counts.get(cat_key, 0.0) + w
                root_counts[root] = root_counts.get(root, 0.0) + w

        max_cat = max(cat_counts.values()) if cat_counts else 0
        max_root = max(root_counts.values()) if root_counts else 0

        preferred_categories = (
            {k: (v / max_cat) for k, v in cat_counts.items()} if max_cat else {}
        )
        preferred_roots = (
            {k: (v / max_root) for k, v in root_counts.items()} if max_root else {}
        )
        
        if len(vvv_purchase_history) == 0:
            # Cold start: Return popular VVV products
            popular_products = sorted(
                adapter.vvv_products, 
                key=lambda x: x.get('popular', 0), 
                reverse=True
            )[:n]
            
            recommendations = [
                {
                    'product_id': int(p['id']),
                    'score': 20.0,
                    'name': p['name'],
                    'price': p['price'],
                    'image': p['image'],
                    'category': f"{p.get('category', '')}/{p.get('subcategory', '')}",
                    'reason': 'Sản phẩm được nhiều người yêu thích'
                }
                for p in popular_products
            ]
            
            return jsonify({
                'user_id': vvv_user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'method': 'popularity'
            })
        
        # 2. Map VVV products → Instacart products
        # Use a recency-boosted proxy list so the mapping reflects the latest orders more strongly.
        proxy_vvv_items = []
        for order in (recent_orders or []):
            dt = _parse_dt(order.get('createdAt') or order.get('created_at'))
            if dt:
                days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
            else:
                days_ago = 0.0
            w_time = math.exp(-days_ago / decay_days)
            # Recent orders (w_time ~ 1) get up to 3x influence; older orders ~ 1x.
            mult = 1 + int(round(2.0 * w_time))

            for item in order.get('items', []) or []:
                pid = item.get('productId')
                if not pid:
                    continue
                try:
                    qty = int(item.get('quantity', 1))
                except Exception:
                    qty = 1
                qty = max(1, min(qty, 50))

                reps = max(1, min(qty * mult, 80))
                proxy_vvv_items.extend([str(pid)] * reps)
                if len(proxy_vvv_items) >= 400:
                    break
            if len(proxy_vvv_items) >= 400:
                break

        instacart_proxy = adapter.vvv_to_instacart_products(proxy_vvv_items or vvv_purchase_history)
        
        # 3. Get recommendations from Instacart model
        # Default to a non-existent user_id so CF branch doesn't bias all users.
        model_user_id = int(data.get('model_user_id', 0))
        instacart_recs = recommender.recommend(
            user_id=model_user_id,
            cart_items=instacart_proxy,
            n=max(50, n * 5),  # Get more to have better choices after mapping
            w_cf=w_cf,
            w_basket=w_basket,
            w_pop=w_pop,
            filter_purchased=False  # Don't filter vì đang map qua VVV
        )
        
        # 4. Map Instacart recommendations → VVV products
        vvv_recommendations = adapter.instacart_to_vvv_products(
            instacart_recs,
            preferred_categories=preferred_categories,
            preferred_roots=preferred_roots,
        )

        # 4.05 Inject VVV-side category personalization so different subcategories diverge clearly.
        # Skip for trending-style requests (high popularity weight).
        is_trending_request = (w_pop >= 0.6 and w_basket <= 0.25)
        purchased_set_all = set(str(x) for x in vvv_purchase_history)

        if (not is_trending_request) and cat_counts:
            # Prefer the top 1-2 most frequent category keys (e.g. veg/root vs veg/leaf)
            top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            boost_base = max((float(r.get('score', 0.0)) for r in vvv_recommendations), default=10.0)

            injected = []
            inject_limit = max(10, n * 3)
            for rank_cat, (cat_key, cnt) in enumerate(top_cats):
                pool = adapter.cat_key_to_products.get(cat_key, [])
                if not pool:
                    continue
                # Create a few candidates from this exact category key
                added = 0
                for i, p in enumerate(pool[:50]):
                    pid = str(p.get('id'))
                    if not pid or pid in purchased_set_all:
                        continue
                    score = boost_base + 6.0 + (2.0 * (cnt / max_cat if max_cat else 0.0)) - (0.02 * i) - (0.2 * rank_cat)
                    injected.append({
                        'product_id': int(pid),
                        'score': round(score, 2),
                        'name': p.get('name'),
                        'price': p.get('price'),
                        'image': p.get('image'),
                        'category': cat_key,
                        'reason': 'Dựa trên những món bạn mua gần đây'
                    })
                    added += 1
                    if added >= 6:
                        break
                    if len(injected) >= inject_limit:
                        break
                if len(injected) >= inject_limit:
                    break

            if injected:
                # Merge injected with ML results (dedupe by product_id, keep higher score)
                merged = {}
                for rec in vvv_recommendations + injected:
                    pid = rec.get('product_id')
                    if pid is None:
                        continue
                    cur = merged.get(pid)
                    if not cur or float(rec.get('score', 0)) > float(cur.get('score', 0)):
                        merged[pid] = rec
                vvv_recommendations = sorted(merged.values(), key=lambda x: x.get('score', 0), reverse=True)

        # 4.1 Re-rank towards user's preferred categories (helps differentiate leaf vs root etc.)
        if vvv_recommendations and (cat_counts or root_counts):
            for rec in vvv_recommendations:
                try:
                    base_score = float(rec.get('score', 0.0))
                except Exception:
                    base_score = 0.0

                cat_key = str(rec.get('category', '')).strip()
                root = cat_key.split('/')[0] if '/' in cat_key else cat_key

                cat_boost = 0.0
                root_boost = 0.0
                if max_cat and cat_key in cat_counts:
                    cat_boost = 0.6 * (cat_counts[cat_key] / max_cat)
                if max_root and root in root_counts:
                    root_boost = 0.3 * (root_counts[root] / max_root)

                # Small penalty for categories the user never bought
                penalty = 0.10 if root and root_counts and root not in root_counts else 0.0
                rec['score'] = round(base_score * (1.0 + cat_boost + root_boost - penalty), 2)

            vvv_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 5. Filter purchased if requested
        if filter_purchased:
            purchased_set = set(vvv_purchase_history)
            vvv_recommendations = [
                rec for rec in vvv_recommendations 
                if str(rec['product_id']) not in purchased_set
            ]
        
        # 6. Return top N
        final_recommendations = vvv_recommendations[:n]
        
        return jsonify({
            'user_id': vvv_user_id,
            'recommendations': final_recommendations,
            'count': len(final_recommendations),
            'method': 'hybrid_instacart_mapping',
            'debug': {
                'vvv_history_count': len(vvv_purchase_history),
                'instacart_proxy_count': len(instacart_proxy),
                'instacart_recs_count': len(instacart_recs),
                'vvv_candidates_count': len(vvv_recommendations)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/similar', methods=['POST'])
def get_similar_items():
    """
    Gợi ý sản phẩm tương tự
    
    Request body:
    {
        "product_id": 24852,
        "n": 10
    }
    
    Response:
    {
        "product_id": 24852,
        "similar_items": [
            {"product_id": 21137, "score": 56156},
            ...
        ]
    }
    """
    try:
        try:
            adapter.reload_if_changed()
        except Exception:
            pass

        data = request.get_json() or {}
        
        if 'product_id' not in data:
            return jsonify({'error': 'product_id is required'}), 400
        
        product_id_raw = data['product_id']
        product_id = int(product_id_raw)
        n = int(data.get('n', 10))

        # If frontend sends a VVV product id, return VVV-enriched similar items.
        vvv_product = adapter.product_index.get(str(product_id))
        if vvv_product:
            # Map VVV product category -> a set of Instacart proxy items
            instacart_seed = adapter.vvv_to_instacart_products([str(product_id)])
            if not instacart_seed:
                return jsonify({
                    'product_id': product_id,
                    'similar_items': [],
                    'count': 0,
                    'method': 'vvv_similar_empty_seed'
                })

            # Aggregate neighbors across seed items
            agg = {}
            for inst_pid in instacart_seed:
                for pid, score in recommender.recommend_similar_items(product_id=inst_pid, n=50):
                    agg[int(pid)] = agg.get(int(pid), 0.0) + float(score)

            ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
            ranked = ranked[: max(50, n * 5)]

            vvv_similar = adapter.instacart_to_vvv_products(ranked)
            # Override reason for "similar" context
            for rec in vvv_similar:
                rec['reason'] = 'Sản phẩm tương tự khẩu vị của bạn'

            vvv_similar = vvv_similar[:n]
            return jsonify({
                'product_id': product_id,
                'similar_items': vvv_similar,
                'count': len(vvv_similar),
                'method': 'vvv_similar'
            })

        # Otherwise treat as Instacart product id (backward-compatible)
        similar = recommender.recommend_similar_items(product_id=product_id, n=n)
        similar_items = [
            {'product_id': int(prod_id), 'score': int(score)}
            for prod_id, score in similar
        ]
        return jsonify({
            'product_id': product_id,
            'similar_items': similar_items,
            'count': len(similar_items),
            'method': 'instacart_similar'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-recommend', methods=['POST'])
def batch_recommendations():
    """
    Gợi ý cho nhiều users cùng lúc (batch processing)
    
    Request body:
    {
        "user_ids": [1, 2, 3, 4, 5],
        "n": 5
    }
    
    Response:
    {
        "results": {
            "1": [{"product_id": 123, "score": 10.5}, ...],
            "2": [...],
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'user_ids' not in data:
            return jsonify({'error': 'user_ids is required'}), 400
        
        user_ids = data['user_ids']
        n = data.get('n', 10)
        
        results = {}
        for user_id in user_ids:
            try:
                recs = recommender.recommend(user_id=int(user_id), n=n)
                results[str(user_id)] = [
                    {'product_id': pid, 'score': round(score, 2)}
                    for pid, score in recs
                ]
            except Exception as e:
                results[str(user_id)] = {'error': str(e)}
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run development server
    print("\n" + "="*50)
    print("🎯 Recommendation API Server")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /api/recommend")
    print("  POST /api/similar")
    print("  POST /api/batch-recommend")
    print("\n" + "="*50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        use_reloader=False
    )
