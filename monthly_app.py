import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from pathlib import Path

# ───────────────────────────────────────────────
# Streamlit Page Config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Monthly BRS Automator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Monthly BRS Automator `v1.1`")
st.markdown("Automated Bank Reconciliation using exact matches, UTR checks, and strict column verification.")

# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def excel_date(serial):
    try: 
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(float(str(serial).strip()), 'D')
    except: 
        return pd.NA

def load_uploaded_file(uploaded_file, find_header_col=None):
    """Loads a CSV or Excel uploaded file, optionally searching for a header row."""
    ext = Path(uploaded_file.name).suffix.lower()
    
    # Read raw to find header
    if ext == '.csv':
        df_full = pd.read_csv(uploaded_file, header=None, dtype=str)
    elif ext in ['.xls', '.xlsx']:
        df_full = pd.read_excel(uploaded_file, header=None, dtype=str)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if find_header_col:
        try:
            # Find the row containing the header column
            matches = df_full.apply(lambda col: col.astype(str).str.contains(find_header_col, case=False, na=False))
            header_row_idx = matches.any(axis=1).idxmax()
            
            uploaded_file.seek(0) # reset pointer
            if ext == '.csv':
                df = pd.read_csv(uploaded_file, skiprows=header_row_idx)
            else:
                df = pd.read_excel(uploaded_file, skiprows=header_row_idx)
                
            return df, header_row_idx
        except Exception as e:
            uploaded_file.seek(0)
            if ext == '.csv':
                return pd.read_csv(uploaded_file), 0
            else:
                return pd.read_excel(uploaded_file), 0
                
    else:
        uploaded_file.seek(0)
        if ext == '.csv':
            return pd.read_csv(uploaded_file), 0
        else:
            return pd.read_excel(uploaded_file), 0


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
# App Interface
# ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📒 General Ledger")
    gl_file = st.file_uploader("Upload GL File (.csv, .xlsx)", type=["csv", "xls", "xlsx"], key="gl")

with col2:
    st.subheader("🏦 Bank Statement")
    bank_file = st.file_uploader("Upload Bank File (.csv, .xlsx)", type=["csv", "xls", "xlsx"], key="bank")

if gl_file and bank_file:
    if st.button("🚀 Run Reconciliation", use_container_width=True, type="primary"):
        
        with st.status("Running Automatic Reconciliation...", expanded=True) as status:
            try:
                st.write("Loading General Ledger dataset...")
                gl, _ = load_uploaded_file(gl_file)
                
                required_gl_cols = ['Amount in local currency', 'Posting Date', 'Vendor Name', 'Assignment', 'Document Number']
                missing_gl_cols = [col for col in required_gl_cols if col not in gl.columns]
                if missing_gl_cols:
                    st.error(f"🚨 Missing required columns in GL file: {', '.join(missing_gl_cols)}\n\nPlease ensure your GL file includes all of the following: {', '.join(required_gl_cols)}")
                    st.stop()
                    
                gl['Amount'] = pd.to_numeric(gl['Amount in local currency'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                gl['Posting Date_DT'] = pd.to_datetime(gl['Posting Date'], dayfirst=True, errors='coerce')

                st.write(f"Loaded {len(gl)} GL records.")

                st.write("Loading Bank Statement dataset...")
                bank, header_row_idx = load_uploaded_file(bank_file, find_header_col='Txn Date')
                bank.columns = [str(c).strip() for c in bank.columns]
                
                required_bank_cols = ['Txn Date', 'Debit', 'Credit', 'Description']
                missing_bank_cols = [col for col in required_bank_cols if col not in bank.columns]
                if missing_bank_cols:
                    st.error(f"🚨 Missing required columns in Bank Statement: {', '.join(missing_bank_cols)}\n\nPlease ensure your Bank Statement includes all of the following: {', '.join(required_bank_cols)}")
                    st.stop()
                    
                bank['Debit_Clean'] = pd.to_numeric(bank['Debit'].astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), errors='coerce').fillna(0)
                bank['Credit_Clean'] = pd.to_numeric(bank['Credit'].astype(str).str.replace(',', '').str.replace('-', '0').str.strip(), errors='coerce').fillna(0)
                bank['Net_Amount'] = bank['Credit_Clean'] - bank['Debit_Clean']

                bank['Txn Date_DT'] = pd.to_datetime(bank['Txn Date'], errors='coerce', dayfirst=True)
                mask_nat = bank['Txn Date_DT'].isna()
                if mask_nat.any():
                    bank.loc[mask_nat, 'Txn Date_DT'] = bank.loc[mask_nat, 'Txn Date'].apply(excel_date)

                st.write(f"Loaded {len(bank)} Bank records (Header detected at row {header_row_idx}).")
                
            except Exception as e:
                st.error(f"Error reading and parsing files: {e}")
                st.stop()

            # Initialize new tracking columns
            bank['Document Number'] = pd.NA
            bank['Assignment'] = pd.NA
            bank['Vendor Name'] = pd.NA
            bank['is_matched'] = False
            bank['Name_Mismatch_Flag'] = ""

            st.write("Starting Global Optimal Matching...")
            
            matched_count = 0
            mismatch_count = 0
            grouped_gl = gl.dropna(subset=['Amount']).groupby('Amount')
            
            # Create a progress bar
            prog_bar = st.progress(0)
            total_groups = len(grouped_gl)
            idx = 0
            
            for amount, group in grouped_gl:
                # Update progress bar
                idx += 1
                prog_bar.progress(idx / total_groups)
                
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
                        
                        if not pair['name_confirmed']:
                            bank.at[pair['b_idx'], 'Name_Mismatch_Flag'] = "Potential Name Mismatch"
                            mismatch_count += 1
                        else:
                            bank.at[pair['b_idx'], 'Name_Mismatch_Flag'] = "Matched"
                            
                        used_gl.add(pair['gl_idx'])
                        used_bank.add(pair['b_idx'])
                        matched_count += 1

            # Prepare final columns
            ext_cols = ['Document Number', 'Assignment', 'Vendor Name', 'Net_Amount', 'Name_Mismatch_Flag']
            original_bank_cols = [c for c in bank.columns if c not in ext_cols and not c.endswith('_Clean') and not c.endswith('_DT') and c != 'is_matched']
            final_columns = original_bank_cols + ext_cols
            final_columns = [col for col in final_columns if col in bank.columns]
            
            final_output = bank[final_columns]
            
            status.update(label=f"Reconciliation Complete! Matched {matched_count} rows.", state="complete", expanded=False)

        # Show Results
        st.success("Reconciliation Complete!")
        
        total_bank_rows = len(bank)
        unmatched_count = total_bank_rows - matched_count
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Bank Rows", total_bank_rows)
        col_m2.metric("Successfully Matched", matched_count)
        col_m3.metric("Name Mismatches", mismatch_count)
        col_m4.metric("Unmatched Rows", unmatched_count)
        
        st.subheader("Preview")
        st.dataframe(final_output.head(20), use_container_width=True)

        # Allow Download
        buf = io.BytesIO()
        final_output.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        
        st.download_button(
            label="⬇️ Download Final Report (.xlsx)",
            data=buf,
            file_name="BRS_Final_With_Mismatch_Flag.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
