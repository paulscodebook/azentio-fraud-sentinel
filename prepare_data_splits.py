#!/usr/bin/env python3
"""
Azentio Hackathon: Deterministic Train / Validation / Test Data Splitter
=======================================================================
Label Provenance Notice:
  The labels generated here are WEAK LABELS produced by deterministic domain
  heuristics. The original challenge datasets (transactions.csv, accounts.csv,
  customers.csv) contain NO ground truth fraud labels. These weak labels serve
  as training supervision and agreement benchmarks, NOT ground-truth reality.

Splits:
  - 70% Train:      700 records (data/train_weak_labels.jsonl)
  - 15% Validation: 150 records (data/val_weak_labels.jsonl)
  - 15% Test:       150 records (data/test_weak_labels.jsonl)
  - Random Seed:    42 (deterministic)
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
from generate_sft_data import create_training_record

def prepare_splits(seed: int = 42):
    base_dir = resolve_base_dir()
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets from {base_dir}...")
    accounts = load_accounts()
    customers = load_customers()
    transactions = load_transactions()

    records = []
    for txn in transactions:
        acc = accounts.get(txn.get("account_id"))
        cust = customers.get(txn.get("customer_id")) if txn.get("customer_id") else None
        fv = feature_vector(txn, acc, cust)
        record = create_training_record(fv)
        record["_provenance"] = "WEAK_LABEL_HEURISTIC"
        records.append(record)

    total = len(records)
    print(f"Total records generated: {total}")

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    train_count = int(total * 0.70)  # 700
    val_count = int(total * 0.15)    # 150
    test_count = total - train_count - val_count  # 150

    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    test_records = [records[i] for i in test_indices]

    train_path = data_dir / "train_weak_labels.jsonl"
    val_path = data_dir / "val_weak_labels.jsonl"
    test_path = data_dir / "test_weak_labels.jsonl"

    def write_jsonl(path, recs):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    def stats(recs):
        fraud_cnt = 0
        for r in recs:
            target = json.loads(r["messages"][2]["content"])
            if target.get("is_fraud"):
                fraud_cnt += 1
        return len(recs), fraud_cnt, len(recs) - fraud_cnt

    print("\nDeterministic Split Statistics (seed=42):")
    tr_tot, tr_f, tr_s = stats(train_records)
    va_tot, va_f, va_s = stats(val_records)
    te_tot, te_f, te_s = stats(test_records)
    print(f"  Train:      {tr_tot} records (Fraud: {tr_f}, Safe: {tr_s}) -> {train_path}")
    print(f"  Validation: {va_tot} records (Fraud: {va_f}, Safe: {va_s}) -> {val_path}")
    print(f"  Test:       {te_tot} records (Fraud: {te_f}, Safe: {te_s}) -> {test_path}")

    return train_path, val_path, test_path

if __name__ == "__main__":
    prepare_splits()
