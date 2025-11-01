import re
from collections import Counter
from flask import render_template_string

def get_analysis_data():
    with open('4d_results_history.csv', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    numbers = re.findall(r'\b\d{4}\b', content)
    numbers = [n for n in numbers if n != '----' and n != '2015' and n != '2025']
    
    freq = Counter(numbers)
    
    # Position frequency
    pos_freq = [{}, {}, {}, {}]
    for num in numbers:
        if len(num) == 4:
            for i, digit in enumerate(num):
                pos_freq[i][digit] = pos_freq[i].get(digit, 0) + 1
    
    # Pattern analysis
    def get_pattern(num):
        if len(set(num)) == 4: return 'ABCD'
        if len(set(num)) == 1: return 'AAAA'
        if len(set(num)) == 2:
            counts = Counter(num)
            return 'AAAB' if 3 in counts.values() else 'AABB'
        return 'AABC'
    
    patterns = Counter([get_pattern(n) for n in numbers if len(n) == 4])
    
    return {
        'hot_numbers': freq.most_common(30),
        'cold_numbers': freq.most_common()[-30:],
        'position_freq': pos_freq,
        'patterns': patterns,
        'total': len(numbers),
        'unique': len(freq)
    }

def analysis_page():
    data = get_analysis_data()
    
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>4D Analysis</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            h1 { color: #333; text-align: center; }
            .section { margin: 30px 0; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .card { background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .number { font-size: 24px; font-weight: bold; color: #2196F3; }
            .count { color: #666; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #4CAF50; color: white; }
            .hot { color: #f44336; }
            .cold { color: #2196F3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎰 4D Lottery Analysis Dashboard</h1>
            
            <div class="section">
                <h2>📊 Summary</h2>
                <div class="grid">
                    <div class="card">
                        <div>Total Numbers</div>
                        <div class="number">{{ total }}</div>
                    </div>
                    <div class="card">
                        <div>Unique Numbers</div>
                        <div class="number">{{ unique }}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="hot">🔥 Top 20 HOT Numbers (Most Frequent)</h2>
                <div class="grid">
                    {% for num, count in hot_numbers[:20] %}
                    <div class="card">
                        <div class="number">{{ num }}</div>
                        <div class="count">{{ count }} times</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2 class="cold">❄️ Top 20 COLD Numbers (Least Frequent - Overdue)</h2>
                <div class="grid">
                    {% for num, count in cold_numbers[:20] %}
                    <div class="card">
                        <div class="number">{{ num }}</div>
                        <div class="count">{{ count }} times</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>📍 Digit Frequency by Position</h2>
                <table>
                    <tr>
                        <th>Position</th>
                        <th>Top 5 Digits</th>
                    </tr>
                    {% for i in range(4) %}
                    <tr>
                        <td>Position {{ i + 1 }}</td>
                        <td>
                            {% for digit, count in position_top[i] %}
                            <span style="margin-right: 15px;">{{ digit }}: {{ count }}</span>
                            {% endfor %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 Pattern Distribution</h2>
                <table>
                    <tr>
                        <th>Pattern</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {% for pattern, count in patterns %}
                    <tr>
                        <td>{{ pattern }}</td>
                        <td>{{ count }}</td>
                        <td>{{ "%.1f"|format((count / total) * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    </body>
    </html>
    '''
    
    position_top = []
    for i in range(4):
        sorted_digits = sorted(data['position_freq'][i].items(), key=lambda x: x[1], reverse=True)[:5]
        position_top.append(sorted_digits)
    
    return render_template_string(html, 
        hot_numbers=data['hot_numbers'],
        cold_numbers=data['cold_numbers'],
        position_top=position_top,
        patterns=data['patterns'].most_common(),
        total=data['total'],
        unique=data['unique']
    )
