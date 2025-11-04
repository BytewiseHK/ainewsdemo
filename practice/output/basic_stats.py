"""
Basic Statistics Analysis for News Articles
Analyzes the trump_xi_meeting_fulltext_dedup-1657.csv file
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def count_words(text):
    """Count words in a text string"""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def main():
    # Define paths
    input_file = "/workspaces/ainewsdemo/data/trump_xi_meeting_fulltext_dedup-1657.csv"
    output_dir = Path("/workspaces/ainewsdemo/practice/output")
    output_csv = output_dir / "trump_xi_meeting_with_wordcount.csv"
    
    print("=" * 60)
    print("Basic Statistics Analysis - News Articles")
    print("=" * 60)
    
    # Load the data
    print("\n1. Loading data...")
    df = pd.read_csv(input_file)
    
    # Count articles
    num_articles = len(df)
    print(f"   Total number of articles: {num_articles}")
    
    # Add word count column
    print("\n2. Calculating word counts for each article...")
    df['word_count'] = df['body'].apply(count_words)
    
    # Save the new CSV with word count
    print(f"   Saving CSV with word count to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Generate statistics
    print("\n3. Generating statistics...")
    print(f"   Average word count: {df['word_count'].mean():.2f}")
    print(f"   Median word count: {df['word_count'].median():.2f}")
    print(f"   Min word count: {df['word_count'].min()}")
    print(f"   Max word count: {df['word_count'].max()}")
    print(f"   Standard deviation: {df['word_count'].std():.2f}")
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    
    # Chart 1: Word Count Distribution (Histogram)
    print("   - Creating histogram of word count distribution...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['word_count'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title('Distribution of Word Counts Across Articles', fontsize=14, fontweight='bold')
    plt.axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["word_count"].mean():.0f}')
    plt.axvline(df['word_count'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["word_count"].median():.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "word_count_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Box Plot
    print("   - Creating box plot...")
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['word_count'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Word Count', fontsize=12)
    plt.title('Box Plot of Word Counts', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "word_count_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Articles by Source (Top 10)
    print("   - Creating top sources chart...")
    if 'source' in df.columns:
        top_sources = df['source'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        top_sources.plot(kind='barh', color='steelblue')
        plt.xlabel('Number of Articles', fontsize=12)
        plt.ylabel('Source', fontsize=12)
        plt.title('Top 10 News Sources by Article Count', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "top_sources.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Chart 4: Sentiment Distribution
    print("   - Creating sentiment distribution chart...")
    if 'sentiment' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(df['sentiment'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel('Sentiment Score', fontsize=12)
        plt.ylabel('Number of Articles', fontsize=12)
        plt.title('Distribution of Sentiment Scores', fontsize=14, fontweight='bold')
        plt.axvline(df['sentiment'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["sentiment"].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sentiment_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Chart 5: Word Count by Language
    print("   - Creating word count by language chart...")
    if 'language' in df.columns:
        lang_stats = df.groupby('language')['word_count'].agg(['mean', 'count'])
        lang_stats = lang_stats[lang_stats['count'] >= 5].sort_values('mean', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(lang_stats)), lang_stats['mean'], color='mediumseagreen', alpha=0.7)
        plt.xlabel('Language', fontsize=12)
        plt.ylabel('Average Word Count', fontsize=12)
        plt.title('Average Word Count by Language (Languages with 5+ articles)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(lang_stats)), lang_stats.index, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "word_count_by_language.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate report
    print("\n5. Generating report...")
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("# News Articles Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Articles**: {num_articles:,}\n")
        f.write(f"- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Source File**: trump_xi_meeting_fulltext_dedup-1657.csv\n\n")
        
        f.write("## Word Count Statistics\n\n")
        f.write(f"- **Average Word Count**: {df['word_count'].mean():.2f}\n")
        f.write(f"- **Median Word Count**: {df['word_count'].median():.2f}\n")
        f.write(f"- **Minimum Word Count**: {df['word_count'].min()}\n")
        f.write(f"- **Maximum Word Count**: {df['word_count'].max()}\n")
        f.write(f"- **Standard Deviation**: {df['word_count'].std():.2f}\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Word Count Distribution\n")
        f.write("![Word Count Distribution](word_count_distribution.png)\n\n")
        f.write("### Box Plot\n")
        f.write("![Box Plot](word_count_boxplot.png)\n\n")
        
        if 'source' in df.columns:
            f.write("### Top News Sources\n")
            f.write("![Top Sources](top_sources.png)\n\n")
        
        if 'sentiment' in df.columns:
            f.write("### Sentiment Distribution\n")
            f.write("![Sentiment Distribution](sentiment_distribution.png)\n\n")
            f.write(f"- **Average Sentiment**: {df['sentiment'].mean():.4f}\n")
            f.write(f"- **Median Sentiment**: {df['sentiment'].median():.4f}\n\n")
        
        if 'language' in df.columns:
            f.write("### Word Count by Language\n")
            f.write("![Word Count by Language](word_count_by_language.png)\n\n")
            f.write("#### Language Distribution\n")
            lang_counts = df['language'].value_counts()
            for lang, count in lang_counts.head(10).items():
                f.write(f"- **{lang}**: {count:,} articles\n")
    
    print(f"   Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutput files generated in: {output_dir}")
    print(f"  - {output_csv.name}")
    print(f"  - word_count_distribution.png")
    print(f"  - word_count_boxplot.png")
    print(f"  - top_sources.png")
    print(f"  - sentiment_distribution.png")
    print(f"  - word_count_by_language.png")
    print(f"  - analysis_report.md")

if __name__ == "__main__":
    main()
