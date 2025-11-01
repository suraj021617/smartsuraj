"""
Structural Motif Clustering
Group draws into clusters based on shared structural traits
"""
from collections import Counter
import numpy as np

def cluster_structural_motifs(df):
    """
    Cluster draws based on gaps, symmetry, and structure
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    draw_features = []
    draw_metadata = []
    
    for idx, row in df_sorted.iterrows():
        date = row['date_parsed']
        draw_id = f"{date.date()}_{row.get('provider', 'unknown')}"
        
        numbers = []
        for col in ['1st_real', '2nd_real', '3rd_real']:
            num = str(row.get(col, ''))
            if num and num.isdigit() and len(num) == 4:
                numbers.append(int(num))
        
        if len(numbers) < 2:
            continue
        
        # Feature 1: Average gap
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        
        # Feature 2: Gap variance (symmetry indicator)
        gap_variance = np.var(gaps) if len(gaps) > 1 else 0
        
        # Feature 3: Digit sum
        digit_sum = sum(int(d) for num in numbers for d in str(num))
        
        draw_features.append([avg_gap, gap_variance, digit_sum])
        draw_metadata.append({'draw_id': draw_id, 'date': date.date(), 'numbers': numbers})
    
    if len(draw_features) < 5:
        return {'clusters': [], 'message': 'Insufficient data'}
    
    # Simple K-means clustering (k=5)
    features_array = np.array(draw_features)
    features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
    
    k = min(5, len(features_array))
    cluster_centers = features_normalized[np.random.choice(len(features_normalized), k, replace=False)]
    
    # Assign clusters
    for _ in range(10):  # 10 iterations
        distances = np.array([[np.linalg.norm(point - center) for center in cluster_centers] 
                              for point in features_normalized])
        labels = np.argmin(distances, axis=1)
        
        for i in range(k):
            cluster_points = features_normalized[labels == i]
            if len(cluster_points) > 0:
                cluster_centers[i] = cluster_points.mean(axis=0)
    
    # Build cluster results
    cluster_data = {i: [] for i in range(k)}
    for idx, label in enumerate(labels):
        cluster_data[label].append(draw_metadata[idx])
    
    clusters = []
    for cluster_id, draws in cluster_data.items():
        if not draws:
            continue
        
        # Representative pattern: most common gap structure
        gap_structures = []
        for draw in draws:
            sorted_nums = sorted(draw['numbers'])
            gaps = tuple(sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1))
            gap_structures.append(gaps)
        
        most_common_gap = Counter(gap_structures).most_common(1)[0][0] if gap_structures else ()
        
        clusters.append({
            'cluster_id': cluster_id,
            'size': len(draws),
            'representative_pattern': str(most_common_gap),
            'recent_draws': [d['draw_id'] for d in draws[-5:]]
        })
    
    clusters.sort(key=lambda x: x['size'], reverse=True)
    return {'clusters': clusters}
