# Fraud Overview

## Executive Summary
This report summarizes the final classification distribution produced by the Fraud Sentinel hybrid pipeline.

*Note: The challenge dataset did not include ground-truth fraud labels. The percentages below represent model and deterministic rule classifications, not classification accuracy.*

## Classification Breakdown
- **Total Transactions Analyzed:** 1,000
- **Transactions Classified as Fraud:** 240 (24.0%)
- **Transactions Classified as Safe:** 760 (76.0%)

## Confidence Distribution (Flagged Fraud)
| Confidence Tier | Record Count | Percentage of Fraud |
|---|---|---|
| 0.90 - 1.00 (Critical) | 49 | 20.4% |
| 0.80 - 0.89 (High) | 25 | 10.4% |
| 0.70 - 0.79 (Elevated) | 140 | 58.3% |
| 0.50 - 0.69 (Moderate) | 26 | 10.8% |

## Top 10 Highest Risk Transactions
| Transaction ID | Confidence | Justification |
|---|---|---|
| [[TXN_0000974]] | 0.97 | Elevated transaction amount of INR 22574.70; Transaction exceeds normal account average by 7.3x. |
| [[TXN_0000394]] | 0.97 | Elevated transaction amount of INR 28370.95; Transaction exceeds normal account average by 6.1x. |
| [[TXN_0000143]] | 0.97 | Elevated transaction amount of INR 38193.90; Cross-border foreign transaction. |
| [[TXN_0000459]] | 0.97 | Missing transaction amount; Transaction exceeds normal account average by 6.7x. |
| [[TXN_0000173]] | 0.97 | Very high transaction amount of INR 110329.47; Transaction amount is 129.3x higher than normal account average. |
| [[TXN_0000455]] | 0.97 | Elevated transaction amount of INR 21429.90; Transaction executed from newly registered device. |
| [[TXN_0000462]] | 0.97 | Very high transaction amount of INR 62146.26; Cross-border foreign transaction. |
| [[TXN_0000179]] | 0.97 | Elevated transaction amount of INR 26978.99; Cross-border foreign transaction. |
| [[TXN_0000443]] | 0.97 | Large negative amount INR -63866.44 indicates unauthorized reversal; Cross-border foreign transaction. |
| [[TXN_0000534]] | 0.97 | Very high transaction amount of INR 59927.08; High-risk merchant category (TANISHQ JEWELLERS). |
