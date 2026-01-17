"""
Test script: Kiểm tra VVV-Instacart adapter và API
Chạy: python test_vvv_integration.py
"""
from pathlib import Path
import sys


# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
# Add project root so we can import as a package: `src.*`
sys.path.insert(0, str(PROJECT_ROOT))

from src.vvv_adapter import VVVInstacartAdapter

def test_adapter():
    """Test VVV adapter"""
    print("="*60)
    print("TEST 1: VVV ADAPTER")
    print("="*60)
    
    VVV_DATA_DIR = PROJECT_ROOT.parent / 'backoffice' / 'data'
    MAPPING_FILE = PROJECT_ROOT / 'mappings' / 'vvv_instacart_mapping.json'
    
    # Initialize
    adapter = VVVInstacartAdapter(VVV_DATA_DIR, MAPPING_FILE)
    
    # Test 1: Get user purchase history
    print("\n[Test 1.1] Get user purchase history")
    for user_id in [4, 5, 6]:
        history = adapter.get_vvv_user_purchase_history(user_id)
        user_info = adapter.get_vvv_user_info(user_id)
        if user_info:
            print(f"  User {user_id} ({user_info['name']}): {len(history)} products purchased")
            print(f"    Products: {history[:5]}")
    
    # Test 2: VVV → Instacart mapping
    print("\n[Test 1.2] VVV → Instacart mapping")
    test_vvv_products = ["100", "110", "200", "300"]
    instacart_mapped = adapter.vvv_to_instacart_products(test_vvv_products)
    print(f"  VVV products: {test_vvv_products}")
    print(f"  Mapped to Instacart: {instacart_mapped[:10]}")
    
    # Test 3: Instacart → VVV mapping
    print("\n[Test 1.3] Instacart → VVV mapping")
    fake_instacart_recs = [
        (13176, 42.5),
        (24852, 38.2),
        (27344, 35.1),
        (38928, 30.5),
        (40706, 28.3)
    ]
    vvv_recs = adapter.instacart_to_vvv_products(fake_instacart_recs)
    print(f"  Instacart recs: {len(fake_instacart_recs)}")
    print(f"  VVV recs: {len(vvv_recs)}")
    print("\n  Top 5 VVV recommendations:")
    for i, rec in enumerate(vvv_recs[:5], 1):
        print(f"    {i}. {rec['name']} (ID: {rec['product_id']}, Score: {rec['score']}) - {rec['reason']}")
    
    print("\n✓ Adapter test completed!\n")
    return adapter


def test_api_simulation(adapter):
    """Simulate API call"""
    print("="*60)
    print("TEST 2: API SIMULATION")
    print("="*60)
    
    from src.recommender import HybridRecommender
    
    # Load Instacart model
    MODELS_DIR = PROJECT_ROOT / "models"
    FEATURES_DIR = PROJECT_ROOT / "data" / "03_features"
    
    print("\n[Test 2.1] Loading Instacart model...")
    recommender = HybridRecommender(MODELS_DIR, FEATURES_DIR)
    print("  ✓ Model loaded")
    
    # Simulate recommendation for VVV user
    print("\n[Test 2.2] Generate recommendations for VVV user")
    # Pick an existing user id from the loaded dataset (demo data may not contain user 4)
    vvv_user_id = int(adapter.vvv_users[0]['id']) if adapter.vvv_users else 0
    
    # Get VVV purchase history
    vvv_history = adapter.get_vvv_user_purchase_history(vvv_user_id)
    user_info = adapter.get_vvv_user_info(vvv_user_id)

    user_name = (user_info or {}).get('name') or 'Unknown'
    print(f"  User: {user_name} (ID: {vvv_user_id})")
    print(f"  Purchase history: {len(vvv_history)} products")
    
    if len(vvv_history) == 0:
        print("  → Cold start: Using popularity")
        popular_products = sorted(
            adapter.vvv_products, 
            key=lambda x: x.get('popular', 0), 
            reverse=True
        )[:10]
        print(f"  Top 5 popular products:")
        for i, p in enumerate(popular_products[:5], 1):
            print(f"    {i}. {p['name']} (popular: {p.get('popular', 0)})")
    else:
        # Map to Instacart
        instacart_proxy = adapter.vvv_to_instacart_products(vvv_history)
        print(f"  Mapped to {len(instacart_proxy)} Instacart products")
        
        # Get Instacart recommendations
        instacart_recs = recommender.recommend(
            user_id=0,
            cart_items=instacart_proxy,
            n=20,
            w_cf=0.2,
            w_basket=0.6,
            w_pop=0.2,
            filter_purchased=False
        )
        print(f"  Instacart recommendations: {len(instacart_recs)}")
        
        # Map back to VVV
        vvv_recs = adapter.instacart_to_vvv_products(instacart_recs)
        print(f"  VVV recommendations: {len(vvv_recs)}")
        
        print(f"\n  Top 10 recommendations for {user_name}:")
        for i, rec in enumerate(vvv_recs[:10], 1):
            print(f"    {i}. {rec['name']}")
            print(f"       Price: {rec['price']:,}đ | Score: {rec['score']} | {rec['reason']}")
    
    print("\n✓ API simulation completed!\n")


def test_multiple_users(adapter):
    """Test cho nhiều users"""
    print("="*60)
    print("TEST 3: MULTIPLE USERS")
    print("="*60)
    
    from src.recommender import HybridRecommender
    
    MODELS_DIR = PROJECT_ROOT / "models"
    FEATURES_DIR = PROJECT_ROOT / "data" / "03_features"
    recommender = HybridRecommender(MODELS_DIR, FEATURES_DIR)
    
    test_users = [4, 5, 6, 7, 8]
    
    for vvv_user_id in test_users:
        user_info = adapter.get_vvv_user_info(vvv_user_id)
        if not user_info:
            continue
        
        print(f"\n[User {vvv_user_id}] {user_info['name']}")
        
        vvv_history = adapter.get_vvv_user_purchase_history(vvv_user_id)
        print(f"  Purchase history: {len(vvv_history)} products")
        
        if len(vvv_history) > 0:
            instacart_proxy = adapter.vvv_to_instacart_products(vvv_history)
            instacart_recs = recommender.recommend(
                user_id=1,
                cart_items=instacart_proxy,
                n=20,
                w_cf=0.2,
                w_basket=0.6,
                w_pop=0.2
            )
            vvv_recs = adapter.instacart_to_vvv_products(instacart_recs)
            
            print(f"  Recommendations: {len(vvv_recs)}")
            print(f"  Top 3:")
            for i, rec in enumerate(vvv_recs[:3], 1):
                print(f"    {i}. {rec['name']} ({rec['score']})")
        else:
            print("  → No history (cold start)")
    
    print("\n✓ Multiple users test completed!\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  VVV-INSTACART INTEGRATION TEST")
    print("="*60 + "\n")
    
    try:
        # Test 1: Adapter
        adapter = test_adapter()
        
        # Test 2: API simulation
        test_api_simulation(adapter)
        
        # Test 3: Multiple users
        test_multiple_users(adapter)
        
        print("="*60)
        print("  ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Hệ thống sẵn sàng! Chạy ML API:")
        print("   cd VuaVuiVe_Recommender")
        print("   python src/api.py")
        print("\n   Sau đó test API:")
        print("   curl -X POST http://localhost:5001/api/recommend \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"user_id\": 4, \"n\": 10}'")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
