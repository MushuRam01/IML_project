import pandas as pd
import os

# Input file
INPUT_FILE = "factor_data.csv"

# Output directory
OUTPUT_DIR = "company_data"

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(INPUT_FILE)

# Ensure Month is datetime
df["Month"] = pd.to_datetime(df["Month"])

# Sort globally first
df = df.sort_values(["co_code", "Month"])

# Create one file per company
for company_code, company_df in df.groupby("co_code"):

    # Sort company entries by time
    company_df = company_df.sort_values("Month")

    # Preserve headers and formatting
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{company_code}.csv"
    )

    # Save file
    company_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

print("Done.")