"""
Flask API cho Recommendation System
Expose endpoints để frontend gọi
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from recommender import HybridRecommender

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS cho frontend gọi được

# Load recommender (chỉ load 1 lần khi start server)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "03_features"

print(" Loading recommendation model...")
recommender = HybridRecommender(MODELS_DIR, FEATURES_DIR)
print(" Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Recommendation API is running'})


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Gợi ý sản phẩm cho user
    
    Request body:
    {
        "user_id": 123,
        "cart_items": [24852, 13176],  // optional
        "n": 10,  // số lượng gợi ý
        "filter_purchased": true  // optional
    }
    
    Response:
    {
        "user_id": 123,
        "recommendations": [
            {"product_id": 27344, "score": 40.28},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if 'user_id' not in data:
            return jsonify({'error': 'user_id is required'}), 400
        
        user_id = int(data['user_id'])
        cart_items = data.get('cart_items', None)
        n = data.get('n', 10)
        filter_purchased = data.get('filter_purchased', True)
        
        # Get recommendations
        recs = recommender.recommend(
            user_id=user_id,
            cart_items=cart_items,
            n=n,
            filter_purchased=filter_purchased
        )
        
        # Format response
        recommendations = [
            {'product_id': prod_id, 'score': round(score, 2)}
            for prod_id, score in recs
        ]
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
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
        data = request.get_json()
        
        if 'product_id' not in data:
            return jsonify({'error': 'product_id is required'}), 400
        
        product_id = int(data['product_id'])
        n = data.get('n', 10)
        
        # Get similar items
        similar = recommender.recommend_similar_items(
            product_id=product_id,
            n=n
        )
        
        # Format response
        similar_items = [
            {'product_id': prod_id, 'score': int(score)}
            for prod_id, score in similar
        ]
        
        return jsonify({
            'product_id': product_id,
            'similar_items': similar_items,
            'count': len(similar_items)
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
        debug=True
    )
