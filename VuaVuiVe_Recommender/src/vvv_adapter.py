"""
VVV-Instacart Adapter
Ánh xạ giữa VVV products/users và Instacart model
"""
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timezone

class VVVInstacartAdapter:
    """Adapter để dùng Instacart model cho VVV data"""
    
    def __init__(self, vvv_data_dir: Path, mapping_file: Path):
        """
        Args:
            vvv_data_dir: Path đến backoffice/data (products.json, orders.json, users.json)
            mapping_file: Path đến vvv_instacart_mapping.json
        """
        self.vvv_data_dir = Path(vvv_data_dir)
        self.mapping_file = Path(mapping_file)

        # Track source file mtimes so we can auto-reload when backoffice sync updates JSON.
        self._data_mtimes: Dict[str, float] = {}
        
        # Load VVV data
        self._load_vvv_data()
        
        # Load mapping
        self._load_mapping()
        
        # Build indexes
        self._build_indexes()
    
    def _load_vvv_data(self):
        """Load VVV products, orders, users"""
        products_path = self.vvv_data_dir / 'products.json'
        orders_path = self.vvv_data_dir / 'orders.json'
        users_path = self.vvv_data_dir / 'users.json'

        with open(products_path, 'r', encoding='utf-8') as f:
            self.vvv_products = json.load(f)
        
        with open(orders_path, 'r', encoding='utf-8') as f:
            self.vvv_orders = json.load(f)
        
        with open(users_path, 'r', encoding='utf-8') as f:
            self.vvv_users = json.load(f)

        # Record mtimes (best-effort; missing files will raise earlier anyway)
        try:
            self._data_mtimes = {
                'products.json': products_path.stat().st_mtime,
                'orders.json': orders_path.stat().st_mtime,
                'users.json': users_path.stat().st_mtime,
            }
        except Exception:
            self._data_mtimes = {}
        
        print(f"✓ Loaded {len(self.vvv_products)} products, {len(self.vvv_orders)} orders, {len(self.vvv_users)} users")

    def reload_if_changed(self) -> bool:
        """Reload VVV data if any source JSON file changed on disk.

        backoffice/server-simple.js syncs db.json -> backoffice/data/*.json.
        This method makes the ML API reflect new orders/products without restarting.

        Returns:
            True if data was reloaded, else False.
        """
        candidates = ['products.json', 'orders.json', 'users.json']
        changed = False
        for filename in candidates:
            path = self.vvv_data_dir / filename
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if self._data_mtimes.get(filename) != mtime:
                changed = True
                break

        if not changed:
            return False

        self._load_vvv_data()
        self._build_indexes()
        return True
    
    def _load_mapping(self):
        """Load category mapping"""
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.category_to_instacart = mapping_data['category_to_instacart']
        
        print(f"✓ Loaded mapping for {len(self.category_to_instacart)} categories")
    
    def _build_indexes(self):
        """Build quick lookup indexes"""
        # Product ID → Product
        self.product_index = {p['id']: p for p in self.vvv_products}
        
        # User ID → Email
        def _norm_str(value: object) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def _norm_email(value: object) -> str:
            return _norm_str(value).lower()

        def _norm_name(value: object) -> str:
            # Keep Vietnamese characters; just normalize whitespace + casefold.
            return " ".join(_norm_str(value).split()).casefold()

        self.user_id_to_email = {u['id']: _norm_email(u.get('email')) for u in self.vvv_users}
        self.email_to_user_id = {self.user_id_to_email[u['id']]: u['id'] for u in self.vvv_users}
        # Name → [user_id] (fallback when order email doesn't match)
        self.name_to_user_ids = {}
        for u in self.vvv_users:
            name_key = _norm_name(u.get('name'))
            if not name_key:
                continue
            self.name_to_user_ids.setdefault(name_key, []).append(u['id'])
        # Deterministic resolution when duplicate names exist
        for name_key in self.name_to_user_ids:
            self.name_to_user_ids[name_key].sort()
        
        # User ID → Orders (best-effort; includes newly placed orders for personalization)
        self.user_orders = {}
        
        for order in self.vvv_orders:
            if not self._is_valid_order(order):
                continue
            
            # Resolve order → user
            email = _norm_email(order.get('email'))
            user_id = self.email_to_user_id.get(email)

            # Fallback: match by customerName when sample data has different emails
            if not user_id:
                name_key = _norm_name(order.get('customerName') or order.get('user', {}).get('name'))
                candidates = self.name_to_user_ids.get(name_key, [])
                if candidates:
                    # Choose smallest id to be deterministic
                    user_id = candidates[0]

            if user_id:
                self.user_orders.setdefault(user_id, []).append(order)

        # Category key → products (normalized) for mapping back from Instacart
        self.cat_key_to_products = {}
        for p in self.vvv_products:
            cat_key = self._get_product_cat_key(p)
            self.cat_key_to_products.setdefault(cat_key, []).append(p)
        for cat_key in self.cat_key_to_products:
            self.cat_key_to_products[cat_key].sort(key=lambda x: x.get('popular', 0), reverse=True)

    def _is_valid_order(self, order: Dict) -> bool:
        """Best-effort filter for orders that should count as personalization history.

        We want recommendations to reflect what the user *just placed*, even before the admin
        marks the order as delivered/paid. So we include early statuses like "placed" and
        payment_status like "pending"/"unpaid", but still exclude cancelled/failed flows.
        """
        order_status = str(order.get('status') or order.get('delivery_status') or '').strip().lower()
        payment_status = str(order.get('payment_status', 'N/A')).strip().lower()

        # Reject obviously invalid orders
        invalid_statuses = {'cancelled', 'canceled', 'failed', 'returned', 'refunded'}
        invalid_payment_statuses = {'failed', 'cancelled', 'canceled'}
        if order_status in invalid_statuses:
            return False
        if payment_status in invalid_payment_statuses:
            return False

        # Must have at least one item
        items = order.get('items')
        if not items:
            return False

        # Accept common in-flight/complete statuses produced by the backoffice normalizer
        valid_statuses = {
            'placed', 'preparing', 'ready',
            'pickup', 'delivering', 'delivered',
            'processing', 'shipped', 'shipping',
            'completed'
        }
        if order_status in valid_statuses:
            return True

        # Accept payment statuses that mean user intent exists (COD/unpaid, VNPAY/pending)
        valid_payment_statuses = {'paid', 'pending', 'unpaid', 'n/a', 'na', 'none', ''}
        return payment_status in valid_payment_statuses

    def _parse_order_dt(self, order: Dict) -> Optional[datetime]:
        """Parse best-effort timestamp for an order and return UTC datetime."""
        raw = (
            order.get('createdAt')
            or order.get('created_at')
            or order.get('created')
            or order.get('created_time')
            or order.get('createdTime')
        )
        if not raw:
            return None
        s = str(raw).strip()
        if not s:
            return None
        try:
            # Support ISO strings like 2026-01-17T03:31:20.525Z
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def get_vvv_orders_by_identity(
        self,
        vvv_user_id: Optional[int] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        limit: Optional[int] = 80,
    ) -> List[Dict]:
        """Return user's orders, sorted newest-first, best-effort.

        This is used to build recency-aware profiles.
        """
        # Fast path: indexed orders (but might be incomplete if identity doesn't match backoffice users.json)
        if vvv_user_id is not None and vvv_user_id in self.user_orders:
            orders = list(self.user_orders.get(vvv_user_id, []))
        else:
            def _norm_str(value: object) -> str:
                if value is None:
                    return ""
                return str(value).strip()

            def _norm_email(value: object) -> str:
                return _norm_str(value).lower()

            def _norm_name(value: object) -> str:
                return " ".join(_norm_str(value).split()).casefold()

            def _norm_phone(value: object) -> str:
                return _norm_str(value)

            target_email = _norm_email(email)
            target_name = _norm_name(name)
            target_phone = _norm_phone(phone)

            orders = []
            for order in self.vvv_orders:
                if not self._is_valid_order(order):
                    continue

                order_email = _norm_email(order.get('email'))
                order_name = _norm_name(order.get('customerName') or order.get('user', {}).get('name'))
                order_phone = _norm_phone(order.get('phone') or order.get('user', {}).get('phone'))

                matched = False
                if target_email and order_email and order_email == target_email:
                    matched = True
                if not matched and target_name and order_name and order_name == target_name:
                    matched = True
                if not matched and target_phone and order_phone and order_phone == target_phone:
                    matched = True
                if not matched:
                    continue
                orders.append(order)

        orders.sort(
            key=lambda o: self._parse_order_dt(o) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        if limit is not None and limit > 0:
            return orders[:limit]
        return orders

    def get_vvv_purchase_history_by_identity(
        self,
        vvv_user_id: Optional[int] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> List[str]:
        """Get purchase history using the most reliable identity available.

        Frontend user IDs (localStorage) often don't match backoffice user IDs.
        So we fall back to scanning orders by email / customerName.
        """
        # 1) Use indexed user_orders when possible
        if vvv_user_id is not None and vvv_user_id in self.user_orders:
            return self.get_vvv_user_purchase_history(vvv_user_id)

        def _norm_str(value: object) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def _norm_email(value: object) -> str:
            return _norm_str(value).lower()

        def _norm_name(value: object) -> str:
            return " ".join(_norm_str(value).split()).casefold()

        def _norm_phone(value: object) -> str:
            # Keep leading zeros; just normalize whitespace.
            return _norm_str(value)

        target_email = _norm_email(email)
        target_name = _norm_name(name)
        target_phone = _norm_phone(phone)

        purchased: List[str] = []
        for order in self.get_vvv_orders_by_identity(
            vvv_user_id=vvv_user_id,
            email=email,
            name=name,
            phone=phone,
            limit=200,
        ):
            for item in order.get('items', []):
                product_id = item.get('productId')
                if not product_id:
                    continue

                # Preserve frequency signal (quantity if available)
                qty = item.get('quantity', 1)
                try:
                    qty_int = int(qty)
                except Exception:
                    qty_int = 1
                qty_int = max(1, min(qty_int, 20))
                purchased.extend([str(product_id)] * qty_int)

        return purchased
    
    def get_vvv_user_purchase_history(self, vvv_user_id: int) -> List[str]:
        """
        Lấy lịch sử mua hàng của VVV user
        CHỈ tính đơn đã thanh toán hoặc đã hoàn thành
        
        Returns:
            List[product_id] - VVV product IDs đã mua
        """
        orders = self.user_orders.get(vvv_user_id, [])
        purchased: List[str] = []
        
        for order in orders:
            if not self._is_valid_order(order):
                continue
            
            for item in order.get('items', []):
                product_id = item.get('productId')
                if not product_id:
                    continue

                qty = item.get('quantity', 1)
                try:
                    qty_int = int(qty)
                except Exception:
                    qty_int = 1
                qty_int = max(1, min(qty_int, 20))
                purchased.extend([str(product_id)] * qty_int)

        return purchased
    
    def vvv_to_instacart_products(self, vvv_product_ids: List[str]) -> List[int]:
        """
        Convert VVV product IDs → Instacart product IDs
        Dựa trên category matching
        
        Args:
            vvv_product_ids: List VVV product IDs (strings như "100", "110")
        
        Returns:
            List[int] - Instacart product IDs
        """
        instacart_products: List[int] = []
        
        for vvv_pid in vvv_product_ids:
            vvv_pid = str(vvv_pid)
            product = self.product_index.get(vvv_pid)
            if not product:
                continue

            cat_key = self._get_product_cat_key(product)
            
            # Map to Instacart products.
            # Previous approach extended the entire category list, which makes many users look similar.
            # New approach picks a small, deterministic subset per VVV product_id to increase personalization.
            instacart_pids = self.category_to_instacart.get(cat_key)
            if not instacart_pids:
                # Fallback: pool all subcategories under the same root (e.g. veg/*)
                root = cat_key.split('/')[0] if '/' in cat_key else cat_key
                pooled: List[int] = []
                for k, ids in self.category_to_instacart.items():
                    if k.startswith(f"{root}/"):
                        pooled.extend(ids)
                instacart_pids = pooled

            if not instacart_pids:
                continue

            picks = self._pick_instacart_ids_for_vvv(vvv_pid, instacart_pids)
            instacart_products.extend(picks)
        
        # Keep duplicates to preserve frequency signal, but cap size to avoid blowups
        if len(instacart_products) > 200:
            instacart_products = instacart_products[:200]
        return instacart_products

    def _pick_instacart_ids_for_vvv(self, vvv_pid: str, instacart_candidates: List[int]) -> List[int]:
        """Pick a deterministic subset of Instacart ids for a given VVV product id.

        This helps different VVV products within the same category map to different Instacart seeds,
        improving personalization without requiring a giant per-product mapping file.
        """
        if not instacart_candidates:
            return []

        # Stable base from product id
        try:
            base = int(str(vvv_pid).strip())
        except Exception:
            base = sum(ord(c) for c in str(vvv_pid))

        uniq = list(dict.fromkeys(int(x) for x in instacart_candidates))
        m = len(uniq)
        if m == 0:
            return []

        # Pick 2 ids by default; if pool is small, pick 1.
        k = 2 if m >= 2 else 1
        step = 7  # a prime-ish step to spread indices
        picks = []
        for i in range(k):
            idx = (base + i * step) % m
            picks.append(uniq[idx])
        return picks

    def _get_product_cat_key(self, product: Dict) -> str:
        """Best-effort category/subcategory key for mapping.

        Some VVV products are missing subcategory or have Vietnamese category labels.
        We infer from image path when possible (e.g. ../images/VEG/leaf/xxx.jpg -> veg/leaf).
        """
        category = str(product.get('category', 'unknown')).strip()
        subcategory = str(product.get('subcategory', 'unknown')).strip()

        # If data is already in expected format
        if category and category != 'unknown' and subcategory and subcategory != 'unknown':
            return f"{category}/{subcategory}"

        # Infer from image path
        image = str(product.get('image', '')).strip()
        normalized = image.replace('\\\\', '/').replace('\\', '/').strip()
        parts = [p for p in normalized.split('/') if p]

        # Find 'images' segment then inspect next two folders
        root_map = {
            'veg': {'veg', 'rau', 'raucu', 'rau củ', 'rau_cu', 'vegetable', 'vegetables', 'veg.'},
            'fruit': {'fruit', 'fruits'},
            'meat': {'meat'},
            'drink': {'drink', 'drinks'},
            'dry': {'dry'},
            'spice': {'spice', 'spices'},
            'household': {'household'},
            'sweet': {'sweet', 'sweets'},
        }
        folder_to_root = {
            'veg': 'veg',
            'vegetable': 'veg',
            'vegetables': 'veg',
            'fruit': 'fruit',
            'meat': 'meat',
            'drink': 'drink',
            'dry': 'dry',
            'spice': 'spice',
            'household': 'household',
            'sweet': 'sweet',
        }

        # Common VVV image folders are uppercased (VEG/FRUIT/...)
        images_idx = -1
        for i, p in enumerate(parts):
            if p.casefold() == 'images':
                images_idx = i
                break

        inferred_root = ''
        inferred_sub = ''
        if images_idx != -1 and images_idx + 2 < len(parts):
            root_folder = parts[images_idx + 1].casefold()
            sub_folder = parts[images_idx + 2].casefold()
            inferred_root = folder_to_root.get(root_folder, root_folder)
            inferred_sub = sub_folder

        # Vietnamese category fallback
        if not inferred_root and category:
            cat_norm = category.strip().casefold()
            # Some records have "Rau củ" as category
            if 'rau' in cat_norm:
                inferred_root = 'veg'

        root_final = (inferred_root or category or 'unknown').strip()
        sub_final = (inferred_sub or subcategory or 'unknown').strip()

        # Ensure lower-case keys for mapping file
        return f"{root_final.casefold()}/{sub_final.casefold()}"
    
    def instacart_to_vvv_products(
        self,
        instacart_recommendations: List[Tuple[int, float]],
        preferred_categories: Optional[Dict[str, float]] = None,
        preferred_roots: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Convert Instacart recommendations → VVV products
        
        Args:
            instacart_recommendations: List[(product_id, score)] từ Instacart model
        
        Returns:
            List[Dict] - VVV products với scores
        """
        # Build reverse mapping: Instacart product ID → VVV categories
        instacart_to_categories = {}
        for cat_key, inst_pids in self.category_to_instacart.items():
            for inst_pid in inst_pids:
                if inst_pid not in instacart_to_categories:
                    instacart_to_categories[inst_pid] = []
                instacart_to_categories[inst_pid].append(cat_key)
        
        # Convert recommendations
        vvv_recommendations = []
        seen_vvv_products = set()
        
        preferred_categories = preferred_categories or {}
        preferred_roots = preferred_roots or {}

        for inst_pid, inst_score in instacart_recommendations:
            # Find matching VVV categories
            matching_categories = instacart_to_categories.get(inst_pid, [])
            
            for cat_key in matching_categories:
                # Find VVV products in this normalized category key
                matching_vvv = self.cat_key_to_products.get(cat_key, [])
                selected_products = self._select_vvv_products_for_instacart(inst_pid, matching_vvv, k=2)

                # Category-aware score adjustment: prioritize user's preferred categories
                root = cat_key.split('/')[0] if '/' in cat_key else cat_key
                cat_pref = float(preferred_categories.get(cat_key, 0.0))
                root_pref = float(preferred_roots.get(root, 0.0))
                penalty = 0.12 if preferred_roots and root and root not in preferred_roots else 0.0

                score_mult = 1.0 + (0.9 * cat_pref) + (0.25 * root_pref) - penalty

                for vvv_product in selected_products:
                    vvv_pid = vvv_product['id']
                    
                    # Skip if already recommended
                    if vvv_pid in seen_vvv_products:
                        continue
                    
                    seen_vvv_products.add(vvv_pid)
                    
                    vvv_recommendations.append({
                        'product_id': int(vvv_pid),
                        'score': round(float(inst_score) * 0.7 * score_mult, 2),  # proxy + personalization
                        'name': vvv_product['name'],
                        'price': vvv_product['price'],
                        'image': vvv_product['image'],
                        'category': cat_key,
                        'reason': self._generate_reason(inst_score)
                    })
        
        # Sort by score and return top results
        vvv_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return vvv_recommendations

    def _select_vvv_products_for_instacart(self, inst_pid: int, products: List[Dict], k: int = 2) -> List[Dict]:
        """Diversify mapping from Instacart signal to VVV products.

        Old behavior always returned top-popular products for that category, making recs look identical.
        We pick from a small top pool but rotate deterministically by inst_pid.
        """
        if not products:
            return []

        top_pool = products[: min(12, len(products))]
        m = len(top_pool)
        if m == 0:
            return []

        k = max(1, min(k, m))
        start = int(inst_pid) % m
        step = 5

        chosen = []
        seen_ids = set()
        for i in range(m):
            idx = (start + i * step) % m
            p = top_pool[idx]
            pid = p.get('id')
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            chosen.append(p)
            if len(chosen) >= k:
                break
        return chosen
    
    def _generate_reason(self, score: float) -> str:
        """Generate human-friendly reason based on score"""
        if score > 35:
            return "Bạn thường mua sản phẩm này"
        elif score > 25:
            return "Khách hàng thường mua cùng"
        elif score > 15:
            return "Phù hợp với sở thích của bạn"
        else:
            return "Sản phẩm được nhiều người yêu thích"
    
    def get_vvv_user_info(self, vvv_user_id: int) -> Optional[Dict]:
        """Get VVV user information"""
        return next((u for u in self.vvv_users if u['id'] == vvv_user_id), None)


# Example usage
if __name__ == '__main__':
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    VVV_DATA_DIR = PROJECT_ROOT.parent / 'backoffice' / 'data'
    MAPPING_FILE = PROJECT_ROOT / 'mappings' / 'vvv_instacart_mapping.json'
    
    # Initialize adapter
    adapter = VVVInstacartAdapter(VVV_DATA_DIR, MAPPING_FILE)
    
    # Test: Get user purchase history
    vvv_user_id = 4  # Nguyễn Văn A
    purchase_history = adapter.get_vvv_user_purchase_history(vvv_user_id)
    print(f"\nUser {vvv_user_id} purchase history: {purchase_history}")
    
    # Test: Map to Instacart
    instacart_products = adapter.vvv_to_instacart_products(purchase_history)
    print(f"Mapped to Instacart products: {instacart_products[:10]}")
    
    # Test: Map back to VVV
    fake_instacart_recs = [(13176, 40.5), (24852, 35.2), (27344, 30.1)]
    vvv_recs = adapter.instacart_to_vvv_products(fake_instacart_recs)
    print(f"\nVVV Recommendations:")
    for rec in vvv_recs[:5]:
        print(f"  - {rec['name']} (score: {rec['score']}) - {rec['reason']}")
