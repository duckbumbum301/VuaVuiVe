"""
Example: Integrate recommendations vào JavaScript frontend
"""

// ====================
// 1. Get recommendations cho user
// ====================
async function getRecommendations(userId, cartItems = null) {
    try {
        const response = await fetch('http://localhost:5001/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                cart_items: cartItems,
                n: 10,
                filter_purchased: true
            })
        });
        
        const data = await response.json();
        return data.recommendations;
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        return [];
    }
}

// Usage:
// const recs = await getRecommendations(123);
// console.log(recs); // [{product_id: 27344, score: 40.28}, ...]


// ====================
// 2. Get similar products
// ====================
async function getSimilarProducts(productId) {
    try {
        const response = await fetch('http://localhost:5001/api/similar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                product_id: productId,
                n: 6
            })
        });
        
        const data = await response.json();
        return data.similar_items;
    } catch (error) {
        console.error('Error fetching similar items:', error);
        return [];
    }
}

// Usage:
// const similar = await getSimilarProducts(24852);


// ====================
// 3. Display recommendations trong HTML
// ====================
async function displayRecommendations(userId, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '<p>Loading recommendations...</p>';
    
    const recommendations = await getRecommendations(userId);
    
    if (recommendations.length === 0) {
        container.innerHTML = '<p>No recommendations available</p>';
        return;
    }
    
    let html = '<div class="recommendations-grid">';
    
    for (const rec of recommendations) {
        // Fetch product details from your products API
        const product = await fetchProductById(rec.product_id);
        
        html += `
            <div class="product-card" data-score="${rec.score}">
                <img src="${product.image}" alt="${product.name}">
                <h3>${product.name}</h3>
                <p class="price">$${product.price}</p>
                <button onclick="addToCart(${rec.product_id})">Add to Cart</button>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}


// ====================
// 4. Real-time recommendations khi thay đổi cart
// ====================
let cartItems = [];

function updateCart(productId, action) {
    if (action === 'add') {
        cartItems.push(productId);
    } else if (action === 'remove') {
        cartItems = cartItems.filter(id => id !== productId);
    }
    
    // Update recommendations based on cart
    refreshRecommendations();
}

async function refreshRecommendations() {
    const userId = getCurrentUserId(); // Your function
    const recs = await getRecommendations(userId, cartItems);
    
    // Update UI
    displayRecommendedProducts(recs);
}


// ====================
// 5. Similar items trên product page
// ====================
async function showSimilarItems(productId) {
    const similarItems = await getSimilarProducts(productId);
    
    const container = document.getElementById('similar-products');
    let html = '<h2>You May Also Like</h2><div class="similar-grid">';
    
    for (const item of similarItems) {
        const product = await fetchProductById(item.product_id);
        html += `
            <div class="product-card">
                <img src="${product.image}" alt="${product.name}">
                <h3>${product.name}</h3>
                <p>$${product.price}</p>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}


// ====================
// 6. Homepage recommendations
// ====================
async function loadHomepageRecommendations() {
    const userId = getCurrentUserId();
    
    if (userId) {
        // Personalized recommendations
        const recs = await getRecommendations(userId);
        displaySection('for-you', recs);
    } else {
        // Popular items (có thể implement endpoint /api/popular)
        displaySection('popular', await getPopularItems());
    }
}


// ====================
// Helper function (example)
// ====================
async function fetchProductById(productId) {
    // Replace with your actual product API
    const response = await fetch(`/api/products/${productId}`);
    return await response.json();
}

function getCurrentUserId() {
    // Get from session/localStorage/cookie
    return localStorage.getItem('user_id') || null;
}

function displayRecommendedProducts(recommendations) {
    // Your UI update logic
    console.log('Updating UI with recommendations:', recommendations);
}

async function getPopularItems() {
    // Implement if needed
    return [];
}

function displaySection(sectionId, products) {
    // Your section display logic
    console.log(`Displaying ${sectionId}:`, products);
}
