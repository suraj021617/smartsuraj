#!/usr/bin/env python3
"""
Quick script to add new winning numbers to CSV
Usage: python add_results.py
"""

import pandas as pd
from datetime import datetime
import os

def add_new_results():
    csv_file = '4d_results_history.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} not found!")
        return
    
    print("🎯 ADD NEW WINNING NUMBERS")
    print("=" * 40)
    
    # Get input
    date = input("📅 Date (YYYY-MM-DD) or press Enter for today: ").strip()
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    provider = input("🏢 Provider (magnum/toto/damacai): ").strip().lower()
    if not provider:
        provider = 'magnum'
    
    print("\n🏆 Enter winning numbers:")
    first = input("1st Prize: ").strip()
    second = input("2nd Prize: ").strip()
    third = input("3rd Prize: ").strip()
    
    if not all([first, second, third]):
        print("❌ All three prizes required!")
        return
    
    # Load existing data
    df = pd.read_csv(csv_file)
    
    # Create new row
    new_row = {
        'date': date,
        'provider': provider,
        '1st': first,
        '2nd': second, 
        '3rd': third,
        '1st_real': first,
        '2nd_real': second,
        '3rd_real': third,
        'special': '',
        'consolation': '',
        'draw_info': f'{provider.title()} Draw'
    }
    
    # Add to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save back to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ Added to {csv_file}:")
    print(f"📅 {date} | 🏢 {provider.title()}")
    print(f"🥇 {first} | 🥈 {second} | 🥉 {third}")
    print("\n🧠 AI will learn from this data for better predictions!")

if __name__ == "__main__":
    add_new_results()