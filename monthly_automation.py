import pandas as pd
import numpy as np
import re
import argparse
import logging
import sys
from pathlib import Path

# ───────────────────────────────────────────────
# Logging Configuration
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BRS_Automator")


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def excel_date(serial):
    try: 
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(float(str(serial).strip()), 'D')
    except: 
        return pd.NA

def load_file(file_path, find_header_col=None):
    """Loads a CSV or Excel file safely, optionally searching for a header row."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.csv':
        df_full = pd.read_csv(file_path, header=None, dtype=str)
    elif ext in ['.xls', '.xlsx']:
        df_full = pd.read_excel(file_path, header=None, dtype=str)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Must be .csv, .xls, or .xlsx")

    if find_header_col:
        try:
            # Find the row containing the header column
            matches = df_full.apply(lambda col: col.astype(str).str.contains(find_header_col, case=False, na=False))
            header_row_idx = matches.any(axis=1).idxmax()
            
            # Reload with correct header
            if ext == '.csv':
                df = pd.read_csv(file_path, skiprows=header_row_idx)
            else:
                df = pd.read_excel(file_path, skiprows=header_row_idx)
                
            return df, header_row_idx
        except Exception as e:
            logger.warning(f"Could not automatically find header row for '{find_header_col}'. Falling back to row 0. Error: {e}")
            if ext == '.csv':
                return pd.read_csv(file_path), 0
            else:
                return pd.read_excel(file_path), 0
                
    else:
        if ext == '.csv':
            return pd.read_csv(file_path), 0
        else:
            return pd.read_excel(file_path), 0


# ───────────────────────────────────────────────
# Core Scoring Algorithm
# ───────────────────────────────────────────────
def get_match_score_final(gl_row, bank_row):
    gl_name = str(gl_row.get('Vendor Name', '')).upper()
    gl_assign = str(gl_row.get('Assignment', '')).upper()
    bank_desc = str(bank_row.get('Description', '')).upper()
    
    score = 0
    # 1. Assignment/UTR Match
    if len(gl_assign) > 5 and gl_assign in bank_desc:
        score += 1000000
    
    # 2. Token Matching
    ignore = {'MR', 'MRS', 'KUMAR', 'KUMA', 'PVT', 'LTD', 'TRANSFER', 'AUTHORITY', 
              'AIRPORTS', 'INDIA', 'TO', 'BY', 'CHQ', 'NEFT', 'RTGS', 'CMP', 
              'SINGH', 'DEVI', 'PRASAD', 'KUMARI'}
              
    gl_tokens = set([w for w in re.findall(r'\b\w+\b', gl_name) if len(w) > 2 and w not in ignore])
    bank_tokens = set([w for w in re.findall(r'\b\w+\b', bank_desc) if len(w) > 2 and w not in ignore])
    
    common = gl_tokens.intersection(bank_tokens)
    score += len(common) * 10000
    
    # 3. Date diff
    if not pd.isna(bank_row.get('Txn Date_DT')) and not pd.isna(gl_row.get('Posting Date_DT')):
        diff = abs((bank_row['Txn Date_DT'] - gl_row['Posting Date_DT']).days)
        score -= min(500, diff)
    
    return score, len(common) > 0 or (len(gl_assign) > 5 and gl_assign in bank_desc)


# ───────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bank Reconciliation Monthly Automation CLI")
    parser.add_argument("--gl", required=True, help="Path to the General Ledger file (.csv, .xls, .xlsx)")
    parser.add_argument("--bank", required=True, help="Path to the Bank Statement file (.csv, .xls, .xlsx)")
    parser.add_argument("--output", default="BRS_Final_With_Mismatch_Flag.xlsx", help="Filename/path for the output Excel sheet (default: BRS_Final_With_Mismatch_Flag.xlsx)")
    
    args = parser.parse_args()

    gl_path = args.gl
    bank_path = args.bank
    output_path = args.output

    if not Path(gl_path).exists():
        logger.error(f"GL file not found: {gl_path}")
        sys.exit(1)
    if not Path(bank_path).exists():
        logger.error(f"Bank Statement file not found: {bank_path}")
        sys.exit(1)

    logger.info("==========================================")
    logger.info("🚀 Starting Monthly BRS Automation [v1.1]")
    logger.info("==========================================")
    logger.info(f"GL File    : {gl_path}")
    logger.info(f"Bank File  : {bank_path}")
    logger.info(f"Output File: {output_path}")

    # 1. Load Data
    try:
        logger.info("Loading General Ledger dataset...")
        gl, _ = load_file(gl_path)
        
        # Clean GL
        required_gl_cols = ['Amount in local currency', 'Posting Date', 'Vendor Name', 'Assignment', 'Document Number']
        missing_gl_cols = [col for col in required_gl_cols if col not in gl.columns]
        if missing_gl_cols:
            logger.error(f"Missing required columns in GL file: {', '.join(missing_gl_cols)}. Please ensure the GL file contains exactly these columns: {', '.join(required_gl_cols)}")
            sys.exit(1)
            
        gl['Amount'] = pd.to_numeric(gl['Amount in local currency'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
        gl['Posting Date_DT'] = pd.to_datetime(gl['Posting Date'], dayfirst=True, errors='coerce')

        logger.info(f"Loaded {len(gl)} GL records.")

        logger.info("Loading Bank Statement dataset...")
        bank, header_row_idx = load_file(bank_path, find_header_col='Txn Date')
        bank.columns = [str(c).strip() for c in bank.columns]
        
        required_bank_cols = ['Txn Date', 'Debit', 'Credit', 'Description']
        missing_bank_cols = [col for col in required_bank_cols if col not in bank.columns]
        if missing_bank_cols:
            logger.error(f"Missing required columns in Bank Statement: {', '.join(missing_bank_cols)}. Please ensure the Bank file contains exactly these columns: {', '.join(required_bank_cols)}")
            sys.exit(1)
            
        bank['Debit_Clean'] = pd.to_numeric(bank['Debit'].astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), errors='coerce').fillna(0)
        bank['Credit_Clean'] = pd.to_numeric(bank['Credit'].astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), errors='coerce').fillna(0)
        bank['Net_Amount'] = bank['Credit_Clean'] - bank['Debit_Clean']

        # Handle float serials or standard dates seamlessly
        bank['Txn Date_DT'] = pd.to_datetime(bank['Txn Date'], errors='coerce', dayfirst=True)
        # Find those that failed standard parsing (possibly excel serials)
        mask_nat = bank['Txn Date_DT'].isna()
        if mask_nat.any():
            bank.loc[mask_nat, 'Txn Date_DT'] = bank.loc[mask_nat, 'Txn Date'].apply(excel_date)

        logger.info(f"Loaded {len(bank)} Bank records (Header detected at row {header_row_idx}).")
        
    except Exception as e:
        logger.error(f"Error reading and parsing files: {e}")
        sys.exit(1)

    # Initialize new tracking columns
    bank['Document Number'] = pd.NA
    bank['Assignment'] = pd.NA
    bank['Vendor Name'] = pd.NA
    bank['is_matched'] = False
    bank['Name_Mismatch_Flag'] = ""

    # 2. Global Optimal Matching
    logger.info("Starting Global Optimal Matching engine...")
    
    matched_count = 0
    mismatch_count = 0
    grouped_gl = gl.dropna(subset=['Amount']).groupby('Amount')
    
    # Convert amounts to rounded integers or strict floats to avoid tiny precision issues,
    # np.isclose handles this, but grouping handles exact floating matches which might miss.
    for amount, group in grouped_gl:
        mask = (np.isclose(bank['Net_Amount'], amount)) & (~bank['is_matched'])
        bank_candidates = bank[mask].copy()
        
        if bank_candidates.empty: 
            continue
        
        all_pairs = []
        for gl_idx, gl_row in group.iterrows():
            for b_idx, b_row in bank_candidates.iterrows():
                score, name_confirmed = get_match_score_final(gl_row, b_row)
                all_pairs.append({
                    'score': score,
                    'name_confirmed': name_confirmed,
                    'gl_idx': gl_idx,
                    'b_idx': b_idx,
                    'gl_row': gl_row
                })
                
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        used_gl = set()
        used_bank = set()
        
        for pair in all_pairs:
            if pair['gl_idx'] not in used_gl and pair['b_idx'] not in used_bank:
                bank.at[pair['b_idx'], 'Document Number'] = pair['gl_row'].get('Document Number', pd.NA)
                bank.at[pair['b_idx'], 'Assignment'] = pair['gl_row'].get('Assignment', pd.NA)
                bank.at[pair['b_idx'], 'Vendor Name'] = pair['gl_row'].get('Vendor Name', pd.NA)
                bank.at[pair['b_idx'], 'is_matched'] = True
                
                # Add Mismatch Flag
                if not pair['name_confirmed']:
                    bank.at[pair['b_idx'], 'Name_Mismatch_Flag'] = "Potential Name Mismatch"
                    mismatch_count += 1
                else:
                    bank.at[pair['b_idx'], 'Name_Mismatch_Flag'] = "Matched"
                    
                used_gl.add(pair['gl_idx'])
                used_bank.add(pair['b_idx'])
                matched_count += 1

    # 3. Final Formatting & Export
    logger.info(f"Reconciliation completed. Total Matched: {matched_count} rows.")
    logger.info(f"Writing final results to: {output_path}")

    # Keep original bank columns + newly added metadata columns
    # We reload original cols gently using the original file and same header logic row
    try:
        orig_bank, _ = load_file(bank_path, find_header_col='Txn Date' if 'Txn Date' in [str(c).strip() for c in bank.columns] else None)
        original_bank_cols = [str(c).strip() for c in orig_bank.columns]
    except:
        # Fallback if original reload fails
        ext_cols = ['Document Number', 'Assignment', 'Vendor Name', 'Net_Amount', 'Name_Mismatch_Flag']
        original_bank_cols = [c for c in bank.columns if c not in ext_cols and not c.endswith('_Clean') and not c.endswith('_DT') and c != 'is_matched']

    final_columns = original_bank_cols + ['Document Number', 'Assignment', 'Vendor Name', 'Net_Amount', 'Name_Mismatch_Flag']
    
    # Handle any columns generated that might drop
    final_columns = [col for col in final_columns if col in bank.columns]
    
    final_output = bank[final_columns]
    
    try:
        final_output.to_excel(output_path, index=False)
        logger.info(f"✅ Success! File generated: {output_path}")
        logger.info(f"⚠️  Number of potential mismatches flagged: {mismatch_count}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
