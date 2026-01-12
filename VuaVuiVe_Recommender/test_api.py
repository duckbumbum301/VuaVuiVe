"""
Test script cho Recommendation API
"""
import requests
import json

API_URL = "http://localhost:5001"

def test_health():
    """Test health check"""
    print("\n" + "="*50)
    print("🔍 Test 1: Health Check")
    print("="*50)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_recommend():
    """Test user recommendations"""
    print("\n" + "="*50)
    print(" Test 2: User Recommendations")
    print("="*50)
    
    payload = {
        "user_id": 1,
        "n": 5,
        "filter_purchased": True
    }
    
    response = requests.post(
        f"{API_URL}/api/recommend",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nUser ID: {result.get('user_id')}")
    print(f"Number of recommendations: {result.get('count')}")
    print("\nTop 5 recommendations:")
    for item in result.get('recommendations', [])[:5]:
        print(f"  - Product {item['product_id']}: score {item['score']}")


def test_recommend_with_cart():
    """Test recommendations with cart items"""
    print("\n" + "="*50)
    print(" Test 3: Recommendations với Cart Items")
    print("="*50)
    
    payload = {
        "user_id": 1,
        "cart_items": [24852, 13176],
        "n": 5
    }
    
    response = requests.post(
        f"{API_URL}/api/recommend",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nCart items: {payload['cart_items']}")
    print("\nRecommendations:")
    for item in result.get('recommendations', []):
        print(f"  - Product {item['product_id']}: score {item['score']}")


def test_similar_items():
    """Test similar items"""
    print("\n" + "="*50)
    print(" Test 4: Similar Items")
    print("="*50)
    
    payload = {
        "product_id": 24852,
        "n": 5
    }
    
    response = requests.post(
        f"{API_URL}/api/similar",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nProduct ID: {result.get('product_id')}")
    print(f"Similar items found: {result.get('count')}")
    print("\nTop 5 similar items:")
    for item in result.get('similar_items', []):
        print(f"  - Product {item['product_id']}: co-occurrence {item['score']}")


def test_batch_recommend():
    """Test batch recommendations"""
    print("\n" + "="*50)
    print(" Test 5: Batch Recommendations")
    print("="*50)
    
    payload = {
        "user_ids": [1, 2, 3],
        "n": 3
    }
    
    response = requests.post(
        f"{API_URL}/api/batch-recommend",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    results = response.json().get('results', {})
    
    for user_id, recs in results.items():
        print(f"\nUser {user_id}:")
        if isinstance(recs, list):
            for item in recs:
                print(f"  - Product {item['product_id']}: score {item['score']}")
        else:
            print(f"  Error: {recs}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Recommendation API")
    print("="*60)
    print("\nMake sure API server is running: python src/api.py")
    print("="*60)
    
    try:
        test_health()
        test_recommend()
        test_recommend_with_cart()
        test_similar_items()
        test_batch_recommend()
        
        print("\n" + "="*60)
        print(" All tests completed!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n Error: Cannot connect to API server!")
        print("Please start the server first: python src/api.py\n")
    except Exception as e:
        print(f"\nError: {e}\n")
