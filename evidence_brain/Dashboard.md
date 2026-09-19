# Fraud Sentinel — Evidence Brain

## Dataset Overview
- **Total Transactions Analyzed:** 1,000
- **Predicted Fraud:** 240 (24.0%)
- **Predicted Safe:** 760 (76.0%)
- **Unique Customers Linked:** 125
- **Unique Accounts Linked:** 186
- **Active Risk Signals:** 12

## Investigation Views
- [[Reports/Fraud Overview|Fraud Overview]] — Global prediction distributions, confidence tiers, and top threats.
- [[Reports/High Risk Transactions|High Risk Transactions]] — Comprehensive ranked directory of all flagged transactions.
- [[Reports/Prompt Injection Cases|Prompt Injection Cases]] — Evaluation findings and security audit of the adversarial firewall.
- [[Reports/Customer Risk|Customer Risk Summary]] — Exposure profiles aggregated by customer identity.
- [[Reports/Account Risk|Account Risk Summary]] — Vulnerability rankings aggregated by account entity.

## Architecture
- **Detection Engine:** The Fraud Sentinel pipeline processes multi-table relational data, neutralizes adversarial injection attempts, and scores fraud risk using high-precision rules and local SLM contextual reasoning.
- **Evidence Brain:** The Evidence Brain is an explainability and investigation layer built on standard Markdown wikilinks. It maps transaction verdicts to underlying accounts, customer histories, and domain risk signals.
- **Independence:** Obsidian is an optional viewer for human investigators and does not participate in or alter the underlying fraud classifications.

## Graph Traversal Path
Reviewers can traverse relationships natively through wikilinks:
```text
Customer  [[CUST_00096]]
   |
   v
Account   [[ACC_000141]]
   |
   v
Transaction [[TXN_0000974]]
   |
   +---> Risk Signal [[High Transaction Amount]]
   +---> Risk Signal [[Elevated Merchant Category Risk]]
   v
Decision: Fraud (Confidence: 0.97)
```
