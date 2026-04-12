"""
Generates sample_bank.csv and sample_gl.csv in the current directory
for testing the BankRecon Pro app.
Run: python generate_sample_data.py
"""
import pandas as pd

bank = pd.DataFrame({
    "Bank Ref":         ["TXN001","TXN002","TXN003","TXN004","TXN005","TXN006","TXN007"],
    "Bank Date":        ["2024-03-01","2024-03-02","2024-03-02","2024-03-03",
                         "2024-03-04","2024-03-05","2024-03-06"],
    "Bank Amount":      [15000.00, 8500.00, 23500.00, 4200.00, 11000.00, 5500.00, 9999.00],
    "Bank Description": [
        "PAYMENT INFOSYS LTD BATCH 01",
        "TATA CONSULTANCY SVCS INV MAR",
        "WIPRO LIMITED MARCH PYMT",        # subset sum: 13500+10000=23500
        "HCL TECHNOLOG 9283 PAY",
        "VENDOR UNKNOWN XYZ CORP",         # no match → unmatched
        "RELIANCE IND LTD PAYMENT",
        "COMPLETELY RANDOM DESC 999",      # description mismatch
    ],
})

gl = pd.DataFrame({
    "GL Vendor Name": ["Infosys Limited","Tata Consultancy Services",
                       "Wipro Limited","Wipro Limited",
                       "HCL Technologies","Reliance Industries","Infosys Limited"],
    "GL Amount":      [15000.00, 8500.00, 13500.00, 10000.00, 4200.00, 5500.00, 9999.00],
    "GL Date":        ["2024-03-01","2024-03-02","2024-03-02","2024-03-02",
                       "2024-03-03","2024-03-05","2024-03-06"],
    "GL Reference":   ["GL-001","GL-002","GL-003","GL-004","GL-005","GL-006","GL-007"],
})

bank.to_csv("sample_bank.csv", index=False)
gl.to_csv("sample_gl.csv", index=False)
print("✅ sample_bank.csv and sample_gl.csv created.")
