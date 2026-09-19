#!/usr/bin/env python3
"""
Azentio Hackathon: SFT Training Data Generator for Tiny SLM (<3B Params)
========================================================================
Generates high-quality domain-specific Supervised Fine-Tuning (SFT) data
from the cleaned relational banking datasets.

Pairs structured transaction risk profiles with expert Chain-of-Thought
justifications and strict JSON target outputs:
{
  "transaction_id": "TXN_xxxxx",
  "is_fraud": bool,
  "confidence": float,
  "justification": "One-sentence plain-text explanation"
}

Output: azentio_fraud_sft_dataset.jsonl
"""

import json
import random
from pathlib import Path
from azentio_fraud_pipeline import (
    load_accounts,
    load_customers,
    load_transactions,
    feature_vector,
    resolve_base_dir,
)

BASE = resolve_base_dir()
OUTPUT_FILE = BASE / "azentio_fraud_sft_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are an expert banking fraud detection AI. "
    "Analyze the provided transaction feature vector and output a single strict JSON object. "
    "Required schema:\n"
    "{\n"
    '  "transaction_id": "<string>",\n'
    '  "is_fraud": <boolean>,\n'
    '  "confidence": <float between 0.0 and 1.0>,\n'
    '  "justification": "<one-sentence plain-text reasoning>"\n'
    "}\n"
    "Do not include markdown code fences or conversational text outside the JSON object."
)

def create_training_record(fv):
    """Convert a feature vector into an instruction-response training sample."""
    tid = fv["transaction_id"]
    amt = fv["amount"]
    amt_str = f"INR {amt:.2f}" if amt is not None else "MISSING"
    merch = fv["merchant_name_clean"] or "UNKNOWN"
    cat_risk = fv["merchant_category_risk"]
    foreign = fv["is_foreign"]
    new_device = fv["is_new_device"]
    txn_24 = fv["txn_24h"]
    ratio = fv["amt_to_avg_ratio"]
    status = fv["account_status"]
    kyc = fv["kyc_status"]
    risk = fv["risk_rating"]
    bal = fv["balance_after"]
    injection = fv.get("injection_detected", False)

    user_content = (
        f"Transaction ID: {tid}\n"
        f"Amount: {amt_str}\n"
        f"Merchant: {merch} (Category Risk: {cat_risk})\n"
        f"Channel: {fv['channel']} | Auth: {fv['auth_method']}\n"
        f"Foreign Transaction: {foreign} | New Device: {new_device}\n"
        f"Hour of Day: {fv['hour']} | Weekend: {fv['is_weekend']}\n"
        f"Distance From Home: {fv['distance_km']} km\n"
        f"24h Txn Velocity: {txn_24} | Amount/Avg Ratio: {ratio}\n"
        f"Account Status: {status} | Balance After: {bal}\n"
        f"Customer Risk Rating: {risk} | KYC Status: {kyc} | PEP: {fv['is_pep']}"
    )

    # Determine ground truth label and justification
    if injection:
        is_fraud = True
        confidence = 0.98
        justification = "Adversarial prompt-injection string detected in payload indicates deliberate evasion attempt."
    elif amt is not None and amt < 0:
        is_fraud = True
        confidence = 0.95
        justification = f"Negative amount ({amt_str}) represents an unauthorized transaction reversal or accounting anomaly."
    elif amt is not None and amt > 25000 and foreign and cat_risk in ("HIGH", "MEDIUM"):
        is_fraud = True
        confidence = 0.92
        justification = f"High-value cross-border transaction in {cat_risk}-risk merchant category ({merch})."
    elif cat_risk == "HIGH" and foreign and new_device:
        is_fraud = True
        confidence = 0.94
        justification = f"High-risk {merch} transaction executed on an unrecognized device from a foreign location."
    elif txn_24 is not None and txn_24 >= 8:
        is_fraud = True
        confidence = 0.91
        justification = f"Severe transaction velocity spike with {txn_24} attempts within a 24-hour window."
    elif status in ("DORMANT", "CLOSED") and amt is not None and amt > 2000:
        is_fraud = True
        confidence = 0.93
        justification = f"Unauthorized high-value activity initiated on an inactive {status} account."
    elif bal is not None and bal < -75000:
        is_fraud = True
        confidence = 0.96
        justification = f"Critical account draining resulting in severe post-transaction deficit of INR {bal:.2f}."
    elif not foreign and not new_device and cat_risk == "LOW" and amt is not None and amt < 6000:
        is_fraud = False
        confidence = 0.92
        justification = "Verified routine domestic purchase executed on an authenticated familiar device."
    elif not foreign and amt is not None and amt < 1000 and fv["channel"] in ("ONLINE", "MOBILE_APP"):
        is_fraud = False
        confidence = 0.89
        justification = "Standard low-value domestic digital transaction consistent with recurring consumer spending."
    else:
        # Borderline evaluation
        risk_score = 0
        factors = []
        if foreign:
            risk_score += 25
            factors.append("cross-border location")
        if new_device:
            risk_score += 20
            factors.append("new device")
        if amt is not None and amt > 15000:
            risk_score += 25
            factors.append("elevated amount")
        if risk == "HIGH":
            risk_score += 20
            factors.append("high-risk customer tier")
        if kyc in ("PENDING", "EXPIRED"):
            risk_score += 15
            factors.append("unverified KYC")

        if risk_score >= 45:
            is_fraud = True
            confidence = round(0.70 + (risk_score - 45) * 0.005, 2)
            justification = f"Multi-signal elevated risk profile flagged due to {', '.join(factors)}."
        else:
            is_fraud = False
            confidence = round(0.75 - risk_score * 0.004, 2)
            justification = "Transaction characteristics fall within acceptable institutional operational tolerances."

    target_json = {
        "transaction_id": tid,
        "is_fraud": is_fraud,
        "confidence": confidence,
        "justification": justification,
    }

    # Format for chat/instruction tuning (OpenAI / ShareGPT / HuggingFace format)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(target_json, indent=2)},
        ]
    }

def main():
    print(f"Generating SFT dataset from {BASE}...")
    accounts = load_accounts()
    customers = load_customers()
    transactions = load_transactions()

    records = []
    for txn in transactions:
        acc = accounts.get(txn.get("account_id"))
        cust = customers.get(txn.get("customer_id")) if txn.get("customer_id") else None
        fv = feature_vector(txn, acc, cust)
        record = create_training_record(fv)
        records.append(record)

    random.seed(42)
    random.shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(records)} SFT training pairs saved to {OUTPUT_FILE}")
    print(f"   Sample Record Output:")
    print(json.dumps(records[0], indent=2)[:400] + "\n...")

if __name__ == "__main__":
    main()
