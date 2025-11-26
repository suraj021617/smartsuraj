"""
Cross-Draw Pattern Linking
Link patterns across providers and prize tiers
"""
from collections import defaultdict, Counter

def link_cross_draw_patterns(df):
    """
    Compare patterns across providers and prize tiers
    """
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    
    provider_patterns = defaultdict(lambda: Counter())  # {provider: {pattern: count}}
    prize_tier_patterns = defaultdict(lambda: Counter())  # {tier: {pattern: count}}
    pattern_migration = defaultdict(list)  # {pattern: [(date, provider)]}
    
    for idx, row in df_sorted.iterrows():
        date = row['date_parsed']
        provider = str(row.get('provider', 'unknown')).lower()
        
        # Analyze each prize tier
        for tier, col in [('1st', '1st_real'), ('2nd', '2nd_real'), ('3rd', '3rd_real')]:
            num = str(row.get(col, ''))
            if not (num and num.isdigit() and len(num) == 4):
                continue
            
            # Pattern: digit uniqueness
            unique_digits = len(set(num))
            pattern = f"unique_{unique_digits}"
            
            provider_patterns[provider][pattern] += 1
            prize_tier_patterns[tier][pattern] += 1
            pattern_migration[pattern].append((date.date(), provider))
            
            # Pattern: even/odd ratio
            evens = sum(1 for d in num if int(d) % 2 == 0)
            eo_pattern = f"{evens}E{4-evens}O"
            provider_patterns[provider][eo_pattern] += 1
            prize_tier_patterns[tier][eo_pattern] += 1
    
    # Cross-provider pattern map
    cross_provider_map = []
    all_patterns = set()
    for patterns in provider_patterns.values():
        all_patterns.update(patterns.keys())
    
    for pattern in all_patterns:
        providers_with_pattern = {prov: provider_patterns[prov][pattern] 
                                  for prov in provider_patterns if provider_patterns[prov][pattern] > 0}
        if len(providers_with_pattern) > 1:
            cross_provider_map.append({
                'pattern': pattern,
                'providers': providers_with_pattern,
                'total_count': sum(providers_with_pattern.values())
            })
    
    cross_provider_map.sort(key=lambda x: x['total_count'], reverse=True)
    
    # Prize tier comparison
    shared_motifs = []
    for pattern in all_patterns:
        tiers_with_pattern = {tier: prize_tier_patterns[tier][pattern]
                              for tier in prize_tier_patterns if prize_tier_patterns[tier][pattern] > 0}
        if len(tiers_with_pattern) == 3:  # Present in all tiers
            shared_motifs.append({
                'pattern': pattern,
                'tiers': tiers_with_pattern
            })
    
    return {
        'cross_provider_map': cross_provider_map[:20],
        'shared_motifs': shared_motifs[:20],
        'provider_patterns': dict(provider_patterns)
    }
