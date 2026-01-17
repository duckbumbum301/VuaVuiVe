"""
Hybrid Recommender System
Kết hợp 3 nguồn: CF (NMF) + Basket (co-occurrence) + Popular
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import load_npz, csr_matrix

class HybridRecommender:
    """Hybrid recommender: CF + Basket + Popular"""

    def __init__(self, models_dir: Path, features_dir: Path):
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        
        # Load model and artifacts
        self._load_all()

    def _load_all(self):
        """Load tất cả artifacts"""
        print("Loading hybrid recommender...")
        
        # 1) NMF model (load từ pickle)
        model_path = self.models_dir / "nmf_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model không tồn tại: {model_path}\n"
                "Hãy train model bằng notebook 03_train_model.ipynb trước!"
            )
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.nmf_model = model_data['model']
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
        
        print(f"  ✓ Loaded NMF model: user_factors {self.user_factors.shape}, item_factors {self.item_factors.shape}")
        
        # 2) CF mappings
        with open(self.features_dir / "cf_mappings.json", "r", encoding="utf-8") as f:
            cf_maps = json.load(f)
            self.user2idx = {int(k): int(v) for k, v in cf_maps["user2idx"].items()}
            self.prod2idx = {int(k): int(v) for k, v in cf_maps["prod2idx"].items()}
            self.idx2user = {int(k): int(v) for k, v in cf_maps["idx2user"].items()}
            self.idx2prod = {int(k): int(v) for k, v in cf_maps["idx2prod"].items()}
        
        # 3) User-item matrix (để filter đã mua)
        self.user_item_matrix = load_npz(self.features_dir / "user_item_matrix.npz")
        
        # 4) Basket neighbors
        with open(self.features_dir / "cooccurrence_neighbors.json", "r", encoding="utf-8") as f:
            neighbors = json.load(f)
            self.basket_neighbors = {int(k): v for k, v in neighbors.items()}
        
        # 5) Popularity
        with open(self.features_dir / "popularity.json", "r", encoding="utf-8") as f:
            pop = json.load(f)
            self.popular_global = pop["global"]
            self.popular_by_dept = {int(k): v for k, v in pop["by_department"].items()}
        
        print(f"✓ Loaded: {len(self.user2idx)} users, {len(self.prod2idx)} products")

    def recommend(
        self,
        user_id: int,
        cart_items: Optional[List[int]] = None,
        n: int = 10,
        w_cf: float = 0.5,
        w_basket: float = 0.3,
        w_pop: float = 0.2,
        filter_purchased: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Gợi ý hybrid cho user.

        Args:
            user_id: ID user
            cart_items: list product_ids trong giỏ (optional)
            n: số lượng gợi ý
            w_cf, w_basket, w_pop: trọng số từng nguồn
            filter_purchased: có loại bỏ sản phẩm đã mua không

        Returns:
            List[(product_id, score)]
        """
        candidates = {}  # product_id -> score

        # 1) CF recommendations (NMF-based)
        if user_id in self.user2idx:
            user_idx = self.user2idx[user_id]
            
            # Check if user_idx trong phạm vi (có thể bị sample)
            if user_idx < self.user_factors.shape[0]:
                # Compute scores: user_vec @ item_factors.T
                user_vec = self.user_factors[user_idx]
                scores_all = user_vec @ self.item_factors.T
                
                # Get top N candidates
                top_indices = np.argsort(scores_all)[::-1][:100]
                
                for item_idx in top_indices:
                    if item_idx >= len(self.idx2prod):
                        continue
                    prod_id = self.idx2prod[int(item_idx)]
                    score = float(scores_all[item_idx])
                    if score > 0:
                        candidates[prod_id] = candidates.get(prod_id, 0) + w_cf * score

        # 2) Basket recommendations
        if cart_items:
            for cart_prod in cart_items:
                if cart_prod in self.basket_neighbors:
                    for neighbor_prod, co_count in self.basket_neighbors[cart_prod][:50]:
                        # Normalize (log scale)
                        score_basket = np.log1p(co_count) / 10.0
                        candidates[neighbor_prod] = candidates.get(neighbor_prod, 0) + w_basket * score_basket

        # 3) Popular fallback
        for prod_id, count in self.popular_global[:50]:
            score_pop = count / 100000.0
            candidates[prod_id] = candidates.get(prod_id, 0) + w_pop * score_pop

        # Filter đã mua (nếu yêu cầu)
        if filter_purchased and user_id in self.user2idx:
            user_idx = self.user2idx[user_id]
            purchased = set(self.user_item_matrix[user_idx].indices)
            purchased_prod_ids = {self.idx2prod[int(i)] for i in purchased if int(i) in self.idx2prod}
            candidates = {pid: sc for pid, sc in candidates.items() if pid not in purchased_prod_ids}

        # Sort and return top N
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def recommend_similar_items(self, product_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Gợi ý sản phẩm tương tự (dựa trên basket co-occurrence).

        Args:
            product_id: ID sản phẩm gốc
            n: số lượng gợi ý

        Returns:
            List[(product_id, score)]
        """
        if product_id not in self.basket_neighbors:
            # Fallback: popular
            return [(pid, float(cnt)) for pid, cnt in self.popular_global[:n]]

        neighbors = self.basket_neighbors[product_id][:n]
        return [(int(pid), float(cnt)) for pid, cnt in neighbors]


def main():
    """Demo usage"""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    features_dir = project_root / "data" / "03_features"

    recommender = HybridRecommender(models_dir, features_dir)

    # Test 1: Gợi ý cho user (không có cart)
    print("\n=== Test 1: Gợi ý cho user_id=1 ===")
    recs = recommender.recommend(user_id=1, n=10)
    for prod_id, score in recs:
        print(f"  product_id={prod_id}, score={score:.4f}")

    # Test 2: Gợi ý cho user (có cart)
    print("\n=== Test 2: Gợi ý cho user_id=1 với cart=[24852, 13176] ===")
    recs = recommender.recommend(user_id=1, cart_items=[24852, 13176], n=10)
    for prod_id, score in recs:
        print(f"  product_id={prod_id}, score={score:.4f}")

    # Test 3: Sản phẩm tương tự
    print("\n=== Test 3: Sản phẩm tương tự với product_id=24852 ===")
    similar = recommender.recommend_similar_items(product_id=24852, n=10)
    for prod_id, score in similar:
        print(f"  product_id={prod_id}, co-occurrence={score:.0f}")


if __name__ == "__main__":
    main()
