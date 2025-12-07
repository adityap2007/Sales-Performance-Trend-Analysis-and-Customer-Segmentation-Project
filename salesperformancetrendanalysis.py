# ==============================================================================
# 0. SETUP: IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. FUNCTION: DATA CLEANING AND FEATURE ENGINEERING
# ==============================================================================
def clean_and_prepare_data(file_path):
    """
    Loads data, performs cleaning (missing values, invalid entries, type conversion),
    and creates the 'Sales' feature.
    """
    print("--- Starting Data Loading and Cleaning ---")
    
    # Load the Data
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None

    # Data Type Conversion
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Handle Missing Values (CustomerID is critical for RFM)
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Filter Invalid Entries (Returns/Bad Data)
    df_cleaned = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

    # Feature Engineering: Calculate Total Sales/Revenue
    df_cleaned['Sales'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
    
    print(f"Data successfully cleaned. Rows remaining: {len(df_cleaned)}")
    return df_cleaned

# ==============================================================================
# 2. FUNCTION: RFM ANALYSIS AND SEGMENTATION
# ==============================================================================
def perform_rfm_analysis(df_cleaned):
    """Calculates RFM metrics, assigns scores and segments, and analyzes segment means."""
    print("\n--- Performing RFM Analysis and Segmentation ---")
    
    # 2a. Calculate R, F, M metrics
    snapshot_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_cleaned.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Sales', 'sum')
    ).reset_index()

    # 2b. Create RFM Scores (using quintiles - 5 groups)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    # Use rank to handle potential ties in Frequency distribution
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

    # Combine Scores and Calculate Total Score
    rfm['RFM_Total_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)

    # 2c. Simple segmentation mapping based on total score
    def segment_customer(score):
        if score >= 13:
            return 'Best Customers'
        elif score >= 10:
            return 'Loyal Customers'
        elif score >= 6:
            return 'Potential/At-Risk'
        else:
            return 'Lost Customers'

    rfm['Segment'] = rfm['RFM_Total_Score'].apply(segment_customer)
    
    # 2d. Analyze Segment Means (New Addition)
    print("\n--- Mean RFM Values by Segment ---")
    segment_means = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary', ascending=False)
    segment_means['Monetary'] = segment_means['Monetary'].map('${:,.2f}'.format)
    print(segment_means)
    
    return rfm

# ==============================================================================
# 3. FUNCTION: VISUALIZATION
# ==============================================================================
def plot_and_save_analysis(df_cleaned, rfm_data):
    """Generates and saves the three required project visualizations."""
    print("\n--- Generating and Saving Visualizations ---")

    # 3a. Monthly Sales Trend
    df_cleaned['InvoiceMonth'] = df_cleaned['InvoiceDate'].dt.to_period('M')
    monthly_sales = df_cleaned.groupby('InvoiceMonth')['Sales'].sum().reset_index()
    monthly_sales['InvoiceMonth'] = monthly_sales['InvoiceMonth'].astype(str)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='InvoiceMonth', y='Sales', data=monthly_sales, marker='o')
    plt.title('Monthly Sales Trend (Revenue Over Time)')
    plt.xlabel('Month')
    plt.ylabel('Total Sales (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('monthly_sales_trend.png')
    plt.close()
    
    print("Saved: monthly_sales_trend.png")

    # 3b. Top Selling Products
    product_sales = df_cleaned.groupby('Description')['Sales'].sum().reset_index()
    top_10_products = product_sales.sort_values(by='Sales', ascending=False).head(10)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Sales', y='Description', data=top_10_products, palette='viridis')
    plt.title('Top 10 Products by Total Sales')
    plt.xlabel('Total Sales (USD)')
    plt.ylabel('Product Description')
    plt.tight_layout()
    plt.savefig('top_10_products.png')
    plt.close()
    
    print("Saved: top_10_products.png")

    # 3c. RFM Segmentation Pie Chart
    segment_counts = rfm_data['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    plt.figure(figsize=(8, 8))
    plt.pie(segment_counts['Count'], labels=segment_counts['Segment'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Customer Segmentation (RFM Distribution)')
    plt.axis('equal')
    plt.savefig('rfm_segmentation_pie.png')
    plt.close()
    
    print("Saved: rfm_segmentation_pie.png")


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    FILE_NAME = 'Online_Retail.csv'

    # 1. Clean and Prepare Data
    clean_data = clean_and_prepare_data(FILE_NAME)
    
    if clean_data is not None:
        # 2. Perform RFM Analysis
        rfm_results = perform_rfm_analysis(clean_data)

        # 3. Visualization and Saving Results
        plot_and_save_analysis(clean_data, rfm_results)
        
        # Save the final RFM table
        rfm_results.to_csv('rfm_segmentation_data.csv', index=False)
        print("\nSaved: rfm_segmentation_data.csv")
        
        print("\n*** Project Workflow Complete! ***")