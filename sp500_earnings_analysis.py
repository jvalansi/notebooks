#!/usr/bin/env python3
"""
S&P 500 Earnings Per Employee Analysis Script

This script collects annual earnings data for S&P 500 companies over the past 10 years,
calculates earnings per employee metrics, and creates comprehensive visualizations.

Data Sources:
- Financial data: Financial Modeling Prep API (annual income statements)
- Employee data: Financial Modeling Prep API (historical employee counts from 10-K filings)

Usage:
    python sp500_earnings_analysis.py [--collect-only | --visualize-only]
    
    --collect-only: Only collect data, skip visualization
    --visualize-only: Only create visualization from existing data
    (no args): Run both collection and visualization
"""

import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import warnings
import requests
import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
TEST_MODE = False  # Set to False for full S&P 500 run
MAX_COMPANIES = 10 if TEST_MODE else None  # Limit companies in test mode
RATE_LIMIT_DELAY = 0.3  # seconds between API calls
MAX_YEARS = 10  # Past 10 years
OUTPUT_FILE = 'data/sp500_annual_earnings_per_employee.csv'
CHECKPOINT_FILE = 'data/sp500_data_checkpoint.csv'
OUTPUT_IMAGE = 'data/sp500_earnings_per_employee_trend.png'
FMP_API_KEY = os.getenv('FMP_KEY')

if not FMP_API_KEY:
    raise ValueError("FMP_KEY not found in environment variables. Please add it to .env file")

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def get_sp500_list():
    """Fetch S&P 500 company list from Wikipedia."""
    print("\n" + "="*80)
    print("PHASE 1: Fetching S&P 500 Company List")
    print("="*80)
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        
        sp500_df = None
        for i, table in enumerate(tables):
            if 'Symbol' in table.columns:
                sp500_df = table
                print(f"  Found S&P 500 table at index {i}")
                break
        
        if sp500_df is None:
            raise ValueError("Could not find S&P 500 constituents table")
        
        print(f"✓ Successfully fetched {len(sp500_df)} companies")
        return sp500_df
    except Exception as e:
        print(f"✗ Error fetching S&P 500 list: {e}")
        raise

def collect_annual_financials(sp500_df, start_index=0):
    """Collect annual financial data for S&P 500 companies using FMP API."""
    print("\n" + "="*80)
    print("PHASE 2: Collecting Annual Financial Data (FMP API)")
    print("="*80)
    if TEST_MODE:
        print(f"🧪 TEST MODE: Limited to {MAX_COMPANIES} companies")
    print(f"Target: {MAX_YEARS} years per company (2015-2024)")
    print(f"Rate limit: {RATE_LIMIT_DELAY}s delay between requests")
    
    annual_results = []
    failed_tickers = []
    
    tickers = sp500_df['Symbol'].tolist()[start_index:]
    if MAX_COMPANIES:
        tickers = tickers[:MAX_COMPANIES]
    
    for idx, ticker in enumerate(tqdm(tickers, desc="Collecting financial data")):
        try:
            company_row = sp500_df[sp500_df['Symbol'] == ticker].iloc[0]
            company_name = company_row.get('Security', ticker)
            sector = company_row.get('GICS Sector', 'Unknown')
            
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&limit=10&apikey={FMP_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                failed_tickers.append(ticker)
                time.sleep(RATE_LIMIT_DELAY)
                continue
            
            for statement in data:
                try:
                    year = statement.get('calendarYear')
                    if year:
                        year = int(year)
                        if 2015 <= year <= 2024:
                            record = {
                                'ticker': ticker,
                                'company': company_name,
                                'sector': sector,
                                'fiscal_year_end': statement.get('date', ''),
                                'year': year,
                                'net_income': statement.get('netIncome'),
                                'revenue': statement.get('revenue'),
                            }
                            annual_results.append(record)
                except Exception as e:
                    print(f"\n  ⚠ Error processing {ticker} year {year}: {e}")
            
            if (idx + 1) % 50 == 0:
                checkpoint_df = pd.DataFrame(annual_results)
                checkpoint_df.to_csv(CHECKPOINT_FILE, index=False)
                print(f"\n  💾 Checkpoint saved: {len(annual_results)} records")
            
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"\n  ✗ Error for {ticker}: {e}")
            failed_tickers.append(ticker)
            time.sleep(RATE_LIMIT_DELAY)
    
    annual_df = pd.DataFrame(annual_results)
    
    print(f"\n✓ Financial data collection complete")
    print(f"  Total records: {len(annual_df)}")
    print(f"  Companies with data: {annual_df['ticker'].nunique()}")
    print(f"  Failed tickers: {len(failed_tickers)}")
    
    return annual_df, failed_tickers

def collect_historical_employees(sp500_df):
    """Collect historical employee counts from FMP API."""
    print("\n" + "="*80)
    print("PHASE 3: Collecting Historical Employee Data (FMP API)")
    print("="*80)
    if TEST_MODE:
        print(f"🧪 TEST MODE: Limited to {MAX_COMPANIES} companies")
    print("Source: SEC 10-K filings via Financial Modeling Prep")
    
    employee_data = []
    failed_tickers = []
    
    tickers = sp500_df['Symbol'].tolist()
    if MAX_COMPANIES:
        tickers = tickers[:MAX_COMPANIES]
    
    for ticker in tqdm(tickers, desc="Collecting employee data"):
        try:
            url = f'https://financialmodelingprep.com/api/v4/historical/employee_count?symbol={ticker}&apikey={FMP_API_KEY}'
            response = requests.get(url)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                for record in data:
                    period = record.get('periodOfReport', '')
                    if period:
                        year = int(period[:4])
                        employee_count = record.get('employeeCount')
                        
                        if employee_count and employee_count > 0:
                            employee_data.append({
                                'ticker': ticker,
                                'year': year,
                                'employees': employee_count,
                                'filing_date': record.get('filingDate'),
                                'period_of_report': period
                            })
            else:
                failed_tickers.append(ticker)
            
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"\n  ✗ Error for {ticker}: {e}")
            failed_tickers.append(ticker)
    
    employee_df = pd.DataFrame(employee_data)
    
    print(f"\n✓ Employee data collection complete")
    print(f"  Total records: {len(employee_df)}")
    print(f"  Companies with employee data: {employee_df['ticker'].nunique()}")
    print(f"  Failed tickers: {len(failed_tickers)}")
    
    return employee_df, failed_tickers

def merge_and_calculate(financial_df, employee_df):
    """Merge financial data with employee data and calculate metrics."""
    print("\n" + "="*80)
    print("PHASE 4: Merging Data and Calculating Metrics")
    print("="*80)
    
    merged_df = financial_df.merge(
        employee_df[['ticker', 'year', 'employees', 'period_of_report']],
        on=['ticker', 'year'],
        how='left'
    )
    
    print(f"  Financial records: {len(financial_df)}")
    print(f"  After merge: {len(merged_df)}")
    print(f"  Records with employee data: {merged_df['employees'].notna().sum()}")
    
    merged_df['earnings_per_employee'] = merged_df['net_income'] / merged_df['employees']
    merged_df['revenue_per_employee'] = merged_df['revenue'] / merged_df['employees']
    
    merged_df['data_quality_flag'] = 'complete'
    merged_df.loc[merged_df['employees'].isna(), 'data_quality_flag'] = 'missing_employees'
    merged_df.loc[merged_df['net_income'].isna(), 'data_quality_flag'] = 'missing_net_income'
    merged_df.loc[(merged_df['employees'].isna()) & (merged_df['net_income'].isna()), 'data_quality_flag'] = 'missing_both'
    
    merged_df = merged_df.sort_values(['ticker', 'year'])
    
    print(f"\n✓ Metrics calculated")
    print(f"  Complete records: {(merged_df['data_quality_flag'] == 'complete').sum()}")
    print(f"  Missing employees: {(merged_df['data_quality_flag'] == 'missing_employees').sum()}")
    
    return merged_df

def validate_data(df):
    """Perform data validation checks."""
    print("\n" + "="*80)
    print("PHASE 5: Data Validation")
    print("="*80)
    
    total_companies = df['ticker'].nunique()
    years_per_company = df.groupby('ticker').size()
    complete_companies = (years_per_company >= 4).sum()
    
    print(f"\n1. Data Completeness:")
    print(f"   Total companies: {total_companies}")
    print(f"   Companies with ≥4 years: {complete_companies} ({complete_companies/total_companies*100:.1f}%)")
    print(f"   Average years per company: {years_per_company.mean():.1f}")
    
    print(f"\n2. Date Range:")
    print(f"   Earliest year: {df['year'].min()}")
    print(f"   Latest year: {df['year'].max()}")
    print(f"   Unique years: {df['year'].nunique()}")
    
    print(f"\n3. Employee Data:")
    print(f"   Records with employee data: {df['employees'].notna().sum()} ({df['employees'].notna().sum()/len(df)*100:.1f}%)")
    if df['employees'].notna().any():
        print(f"   Employee range: {df['employees'].min():.0f} - {df['employees'].max():.0f}")
        print(f"   Median employees: {df['employees'].median():.0f}")

def export_data(df, output_file):
    """Export final dataset to CSV."""
    print("\n" + "="*80)
    print("PHASE 6: Exporting Data")
    print("="*80)
    
    column_order = [
        'ticker', 'company', 'sector', 'fiscal_year_end', 'year',
        'net_income', 'revenue', 'employees', 
        'earnings_per_employee', 'revenue_per_employee',
        'period_of_report', 'data_quality_flag'
    ]
    
    df_export = df[[col for col in column_order if col in df.columns]].copy()
    df_export.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file)
    
    print(f"✓ Data exported to: {output_file}")
    print(f"  Total records: {len(df_export)}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def load_data(filename):
    """Load the S&P 500 earnings data."""
    print(f"\nLoading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"✓ Loaded {len(df)} records")
    
    df = df[(df['year'] >= 2015) & (df['year'] <= 2024)]
    print(f"✓ Filtered to include completed fiscal years 2015-2024")
    print(f"  Records after filtering: {len(df)}")
    print(f"  Companies: {df['ticker'].nunique()}")
    print(f"  Years: {sorted(df['year'].unique())}")
    return df

def calculate_yearly_averages(df):
    """Calculate aggregate earnings per employee by year."""
    print("\nCalculating yearly aggregates...")
    
    complete_data = df[df['data_quality_flag'] == 'complete'].copy()
    print(f"  Records with complete data: {len(complete_data)}")
    
    yearly_stats = complete_data.groupby('year').agg({
        'earnings_per_employee': ['mean', 'median', 'std', 'count'],
        'ticker': 'nunique',
        'employees': 'sum',
        'net_income': 'sum'
    }).round(0)
    
    yearly_stats.columns = ['mean_earnings_per_employee', 'median_earnings_per_employee',
                           'std_earnings_per_employee', 'company_count', 'unique_companies',
                           'total_employees', 'total_net_income']
    
    yearly_stats['aggregate_earnings_per_employee'] = (
        yearly_stats['total_net_income'] / yearly_stats['total_employees']
    )
    
    yearly_stats['yoy_growth_pct'] = yearly_stats['aggregate_earnings_per_employee'].pct_change() * 100
    
    print("\nYearly Statistics:")
    print(yearly_stats.to_string())
    
    return yearly_stats

def create_visualization(yearly_stats):
    """Create comprehensive visualization of earnings per employee trends."""
    print("\nCreating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('S&P 500 Comprehensive Analysis (2015-2024)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    years = yearly_stats.index
    aggregate_earnings = yearly_stats['aggregate_earnings_per_employee']
    
    # Plot 1: Earnings per Employee Trend
    ax1.plot(years, aggregate_earnings, marker='o', linewidth=2.5,
             markersize=10, color='#2E86AB', label='Aggregate Earnings per Employee')
    
    z = np.polyfit(years, aggregate_earnings, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), "--", color='#A23B72', linewidth=2,
             alpha=0.7, label=f'Trend Line (slope: ${z[0]:,.0f}/year)')
    
    for year, earnings in zip(years, aggregate_earnings):
        ax1.annotate(f'${earnings/1000:.0f}K',
                    xy=(year, earnings),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    fontweight='bold')
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Aggregate Earnings per Employee ($)', fontsize=12, fontweight='bold')
    ax1.set_title('S&P 500 Aggregate Earnings per Employee (2015-2024)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Year-over-Year Growth Rate
    growth_years = yearly_stats.index[1:]
    growth_rates = yearly_stats['yoy_growth_pct'].iloc[1:]
    
    colors = ['#06A77D' if x >= 0 else '#D62246' for x in growth_rates]
    bars = ax2.bar(growth_years, growth_rates, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, rate in zip(bars, growth_rates):
        height = bar.get_height()
        ax2.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Year-over-Year Growth (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Year-over-Year Growth Rate', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Total Employee Count
    total_employees = yearly_stats['total_employees']
    bars3 = ax3.bar(years, total_employees / 1_000_000, color='#F18F01', alpha=0.7, edgecolor='black')
    
    for bar, employees in zip(bars3, total_employees):
        height = bar.get_height()
        ax3.annotate(f'{employees/1_000_000:.1f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Employees (Millions)', fontsize=12, fontweight='bold')
    ax3.set_title('S&P 500 Total Employee Count (2015-2024)',
                  fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}M'))
    
    # Plot 4: Total Net Income
    total_income = yearly_stats['total_net_income']
    bars4 = ax4.bar(years, total_income / 1_000_000_000, color='#06A77D', alpha=0.7, edgecolor='black')
    
    for bar, income in zip(bars4, total_income):
        height = bar.get_height()
        ax4.annotate(f'${income/1_000_000_000:.1f}B',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total Net Income (Billions)', fontsize=12, fontweight='bold')
    ax4.set_title('S&P 500 Total Net Income (2015-2024)',
                  fontsize=13, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}B'))
    
    fig.text(0.5, 0.02, 'Note: Analysis covers 10 years of complete fiscal year data (2015-2024).',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {OUTPUT_IMAGE}")
    plt.show()
    
    return fig

def print_summary_statistics(yearly_stats, df):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n1. Overall Metrics (All Years):")
    print(f"   Aggregate Earnings per Employee: ${yearly_stats['aggregate_earnings_per_employee'].mean():,.0f}")
    print(f"   Minimum (by year): ${yearly_stats['aggregate_earnings_per_employee'].min():,.0f}")
    print(f"   Maximum (by year): ${yearly_stats['aggregate_earnings_per_employee'].max():,.0f}")
    
    print("\n2. Growth Analysis:")
    avg_growth = yearly_stats['yoy_growth_pct'].mean()
    print(f"   Average YoY Growth: {avg_growth:.2f}%")
    print(f"   Best Year: {yearly_stats['yoy_growth_pct'].idxmax()} ({yearly_stats['yoy_growth_pct'].max():.2f}%)")
    print(f"   Worst Year: {yearly_stats['yoy_growth_pct'].idxmin()} ({yearly_stats['yoy_growth_pct'].min():.2f}%)")
    
    years = yearly_stats.index.values
    earnings = yearly_stats['aggregate_earnings_per_employee'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, earnings)
    
    print("\n3. Trend Analysis:")
    print(f"   Linear Trend Slope: ${slope:,.0f} per year")
    print(f"   R-squared: {r_value**2:.4f}")
    print(f"   10-Year Total Change (2015-2024): ${earnings[-1] - earnings[0]:,.0f} ({((earnings[-1]/earnings[0])-1)*100:.1f}%)")
    
    complete_data = df[df['data_quality_flag'] == 'complete']
    print("\n4. Data Quality:")
    print(f"   Total Records: {len(df)}")
    print(f"   Complete Records: {len(complete_data)} ({len(complete_data)/len(df)*100:.1f}%)")
    print(f"   Companies Analyzed: {complete_data['ticker'].nunique()}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def collect_data():
    """Run data collection pipeline."""
    print("\n" + "="*80)
    print("S&P 500 EARNINGS PER EMPLOYEE DATA COLLECTION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    sp500_df = get_sp500_list()
    
    if TEST_MODE:
        print(f"\n🧪 TEST MODE ENABLED: Processing only first {MAX_COMPANIES} companies")
    
    financial_df, failed_financial = collect_annual_financials(sp500_df)
    employee_df, failed_employee = collect_historical_employees(sp500_df)
    final_df = merge_and_calculate(financial_df, employee_df)
    validate_data(final_df)
    export_data(final_df, OUTPUT_FILE)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time/60:.1f} minutes")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(final_df)}")
    print(f"Companies: {final_df['ticker'].nunique()}")

def visualize_data():
    """Run visualization pipeline."""
    print("\n" + "="*80)
    print("S&P 500 EARNINGS PER EMPLOYEE VISUALIZATION")
    print("="*80)
    
    df = load_data(OUTPUT_FILE)
    yearly_stats = calculate_yearly_averages(df)
    create_visualization(yearly_stats)
    print_summary_statistics(yearly_stats, df)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output saved to: {OUTPUT_IMAGE}")

def main():
    """Main execution function."""
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == '--collect-only':
                collect_data()
            elif sys.argv[1] == '--visualize-only':
                visualize_data()
            else:
                print("Usage: python sp500_earnings_analysis.py [--collect-only | --visualize-only]")
                return 1
        else:
            # Run both collection and visualization
            collect_data()
            visualize_data()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())