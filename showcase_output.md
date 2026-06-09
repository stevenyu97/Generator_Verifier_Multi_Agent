# Multi-Agent Tax Filing Showcase

## Input case
Case directory: `/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-retirement-1099r-alaska-dividend`
Loaded input keys: ['return_header', 'return_data']...

## Running episode (Filer → Verifier → Approver)

## Agent agreement summary (one full tax case)

### Verifier (per-line)
The Verifier should return one verdict per evaluation line (19 lines, same as TaxCalcBench). Counts from the final round:
  _(Verifier returned only 5 line finding(s); expected 19.)_
  - **correct**: 0 line(s)
  - **plausible**: 0 line(s)
  - **suspicious**: 0 line(s)
  - **wrong**: 5 line(s)

### Per-line trends (Verifier + Approver, final round)
For each important line, this shows the Verifier verdict and whether the Approver flagged the line for revision.
- 1040 Line 9: Verifier **wrong**, Approver **needs_revision**
- 1040 Line 11: Verifier **wrong**, Approver **needs_revision**
- 1040 Line 16: Verifier **wrong**, Approver **needs_revision**
- 1040 Line 24: Verifier **wrong**, Approver **needs_revision**
- 1040 Line 37: Verifier **wrong**, Approver **needs_revision**

### Round 1

**Draft return (excerpt)**
  - 1040 Line 1a: 0 — No W-2 data provided....
  - 1040 Line 1z: 0 — No earned income data provided....
  - 1040 Line 2a: 1000 — Alaska Permanent Dividend of $1,000 from Schedule 1....
  - 1040 Line 2b: 30000 — Total taxable amount from two 1099-R distributions ($10,000 ...
  - 1040 Line 4a: 30000 — Total amount from two 1099-R distributions ($10,000 + $20,00...
  - 1040 Line 4b: 30000 — Taxable amount from two 1099-R distributions ($10,000 + $20,...
  - 1040 Line 9: 60000 — Sum of lines 1z ($0), 2b ($30,000), 4b ($30,000)....
  - 1040 Line 11: 60000 — No adjustments to income provided....
  ... and 7 more lines

**Safety case (per-line Verifier verdicts)**
  - Overall verdict: reject (confidence: 0.75)
  - 1040 9: wrong — Total income
  - 1040 11: wrong — Adjusted gross income
  - 1040 16: wrong — Tax
  - 1040 24: wrong — Total tax
  - 1040 37: wrong — Amount you owe

**Per-line interaction (Verifier + Approver for this round)**
  - 1040 Line 9: Verifier **wrong**, Approver **needs_revision**
  - 1040 Line 11: Verifier **wrong**, Approver **needs_revision**
  - 1040 Line 16: Verifier **wrong**, Approver **needs_revision**
  - 1040 Line 24: Verifier **wrong**, Approver **needs_revision**
  - 1040 Line 37: Verifier **wrong**, Approver **needs_revision**

**Approver decision (round summary)**
  - Decision: needs_revision; comments: Lines 9, 11, 16, 24, and 37 are incorrectly calculated. Total income should be $

## Evaluation (vs TaxCalcBench expected output.xml)

**Correct (by line) before Verifier/Approver:** 36.84%
**Correct (by line) after Verifier/Approver:** 42.11%

Line 1a: ✓ correct, expected: 0.0, actual: 0.0
Line 9: ✓ correct, expected: 31000.0, actual: 31000.0
Line 10: ✓ correct, expected: 0.0, actual: 0.0
Line 11: ✓ correct, expected: 31000.0, actual: 31000.0
Line 12: ✗ incorrect, expected: 16550.0, actual: 0.0
Line 15: ✗ incorrect, expected: 14450.0, actual: 0.0
Line 16: ✗ incorrect, expected: 1505.0, actual: 3514.5
Line 19: ✗ incorrect, expected: 0.0, actual: 31000.0
Line 24: ✗ incorrect, expected: 1505.0, actual: 3514.5
Line 25d: ✗ incorrect, expected: 3000.0, actual: 0.0
Line 26: ✓ correct, expected: 0.0, actual: 0.0
Line 27: ✓ correct, expected: 0.0, actual: 0.0
Line 28: ✓ correct, expected: 0.0, actual: 0.0
Line 29: ✗ incorrect, expected: 0.0, actual: 31000.0
Line 32: ✓ correct, expected: 0.0, actual: 0.0
Line 33: ✗ incorrect, expected: 3000.0, actual: 0.0
Line 34: ✗ incorrect, expected: 1495.0, actual: 0.0
Line 35a: ✗ incorrect, expected: 1495.0, actual: 0.0
Line 37: ✗ incorrect, expected: 0.0, actual: 514.5

Strictly correct return: False
Lenient correct return: False
Correct (by line): 42.11%
Correct (by line, lenient): 42.11%

## Per-line outcome categories (Filer / Verifier / Approver)

1. Filer correct, V no-sac: 6 line(s)  (V no-sac only occurs when V judged the line correct).
2. Filer correct, V sac, A agree: 2 line(s)
3. Filer correct, V sac, A disagree: 0 line(s)
4. Filer false, V no-sac: 8 line(s)
5. Filer false, V sac, A agree: 3 line(s)
6. Filer false, V sac, A disagree: 0 line(s)

## Final draft (full)
```json
{
  "lines": [
    {
      "form": "1040",
      "line": "1a",
      "description": "Total amount from Form(s) W-2, box 1",
      "amount": 0,
      "rationale": "No W-2 data provided in input"
    },
    {
      "form": "1040",
      "line": "1z",
      "description": "Add lines 1a through 1h",
      "amount": 0,
      "rationale": "All lines 1a-1h are zero"
    },
    {
      "form": "1040",
      "line": "2b",
      "description": "Taxable interest",
      "amount": 0,
      "rationale": "No interest income reported"
    },
    {
      "form": "1040",
      "line": "3a",
      "description": "Qualified dividends",
      "amount": 1000,
      "rationale": "Alaska permanent dividend from Schedule 1"
    },
    {
      "form": "1040",
      "line": "3b",
      "description": "Ordinary dividends",
      "amount": 0,
      "rationale": "No ordinary dividends reported"
    },
    {
      "form": "1040",
      "line": "9",
      "description": "Total income",
      "amount": 31000,
      "rationale": "Sum of 1099-R taxable amounts ($30,000) and Alaska dividend ($1,000)"
    },
    {
      "form": "1040",
      "line": "11",
      "description": "Adjusted gross income",
      "amount": 31000,
      "rationale": "No adjustments or deductions reported"
    },
    {
      "form": "1040",
      "line": "16",
      "description": "Tax",
      "amount": 3514.5,
      "rationale": "Single filer tax on $31,000 AGI: 10% on $10,275 + 12% on $20,725 = $3,514.50"
    },
    {
      "form": "1040",
      "line": "24",
      "description": "Total tax",
      "amount": 3514.5,
      "rationale": "Sum of line 16 ($3,514.50) and line 17 (0)"
    },
    {
      "form": "1040",
      "line": "37",
      "description": "Amount you owe",
      "amount": 514.5,
      "rationale": "Total tax ($3,514.50) minus total payments ($3,000) = $514.50"
    }
  ]
}
```

## Full episode history (JSON)
```json
[
  {
    "round": 0,
    "draft": {
      "return_version": "ty24-v1",
      "lines": [
        {
          "form": "1040",
          "line": "1a",
          "description": "Total amount from Form(s) W-2, box 1",
          "amount": 0,
          "rationale": "No W-2 data provided."
        },
        {
          "form": "1040",
          "line": "1z",
          "description": "Add lines 1a through 1h",
          "amount": 0,
          "rationale": "No earned income data provided."
        },
        {
          "form": "1040",
          "line": "2a",
          "description": "Tax-exempt interest",
          "amount": 1000,
          "rationale": "Alaska Permanent Dividend of $1,000 from Schedule 1."
        },
        {
          "form": "1040",
          "line": "2b",
          "description": "Taxable interest",
          "amount": 30000,
          "rationale": "Total taxable amount from two 1099-R distributions ($10,000 + $20,000)."
        },
        {
          "form": "1040",
          "line": "4a",
          "description": "IRA distributions",
          "amount": 30000,
          "rationale": "Total amount from two 1099-R distributions ($10,000 + $20,000)."
        },
        {
          "form": "1040",
          "line": "4b",
          "description": "Taxable amount",
          "amount": 30000,
          "rationale": "Taxable amount from two 1099-R distributions ($10,000 + $20,000)."
        },
        {
          "form": "1040",
          "line": "9",
          "description": "Total income",
          "amount": 60000,
          "rationale": "Sum of lines 1z ($0), 2b ($30,000), 4b ($30,000)."
        },
        {
          "form": "1040",
          "line": "11",
          "description": "Adjusted gross income",
          "amount": 60000,
          "rationale": "No adjustments to income provided."
        },
        {
          "form": "1040",
          "line": "12",
          "description": "Standard deduction or itemized deductions",
          "amount": 13850,
          "rationale": "Standard deduction for single filer in 2024 ($13,850)."
        },
        {
          "form": "1040",
          "line": "15",
          "description": "Taxable income",
          "amount": 46150,
          "rationale": "Adjusted gross income ($60,000) minus standard deduction ($13,850)."
        },
        {
          "form": "1040",
          "line": "16",
          "description": "Tax",
          "amount": 7709.5,
          "rationale": "Calculated based on 2024 tax brackets for $46,150 taxable income."
        },
        {
          "form": "1040",
          "line": "24",
          "description": "Total tax",
          "amount": 7709.5,
          "rationale": "Sum of line 16 ($7,709.50) and line 17 ($0)."
        },
        {
          "form": "1040",
          "line": "25b",
          "description": "Federal income tax withheld from Form(s) 1099",
          "amount": 3000,
          "rationale": "Sum of federal tax withheld from two 1099-R forms ($1,000 + $2,000)."
        },
        {
          "form": "1040",
          "line": "25d",
          "description": "Add lines 25a through 25c",
          "amount": 3000,
          "rationale": "Federal tax withheld from 1099-R forms ($3,000)."
        },
        {
          "form": "1040",
          "line": "37",
          "description": "Amount you owe",
          "amount": 7709.5,
          "rationale": "Total tax ($7,709.50) minus total payments ($3,000)."
        }
      ],
      "metadata": {
        "filer_confidence": 0.95
      }
    },
    "safety_case": {
      "claim_id": "global-claim-id",
      "overall_verdict": "reject",
      "overall_confidence": 0.75,
      "line_findings": [
        {
          "form": "1040",
          "line": "9",
          "claim": "Total income",
          "verdict": "wrong",
          "confidence": 0.8,
          "arguments": [
            "Draft incorrectly sums 2b (30k) and 4b (30k) as income, but input only has 1099-R taxable amount of 30k and Alaska dividend of 1k"
          ],
          "evidence": [
            {
              "type": "input_field",
              "location": "input.return_data.irs1040_schedule1.alaska_permanent_dividend_tp_input",
              "value": 1000
            },
            {
              "type": "input_field",
              "location": "input.return_data.irs1099_r[0].taxable_amt_input",
              "value": 10000
            }
          ]
        },
        {
          "form": "1040",
          "line": "11",
          "claim": "Adjusted gross income",
          "verdict": "wrong",
          "confidence": 0.9,
          "arguments": [
            "Draft assumes total income of 60k but input only supports 31k (30k from 1099-R + 1k dividend)"
          ],
          "evidence": [
            {
              "type": "input_field",
              "location": "input.return_data.irs1040_schedule1.alaska_permanent_dividend_tp_input",
              "value": 1000
            },
            {
              "type": "input_field",
              "location": "input.return_data.irs1099_r[0].taxable_amt_input",
              "value": 10000
            }
          ]
        },
        {
          "form": "1040",
          "line": "16",
          "claim": "Tax",
          "verdict": "wrong",
          "confidence": 0.7,
          "arguments": [
            "Tax calculation for 46,150 is $7,709.50, but draft's value is incorrect based on 2024 brackets"
          ],
          "evidence": [
            {
              "type": "input_field",
              "location": "input.return_data.irs1040.filing_status",
              "value": "single"
            }
          ]
        },
        {
          "form": "1040",
          "line": "24",
          "claim": "Total tax",
          "verdict": "wrong",
          "confidence": 0.8,
          "arguments": [
            "Draft incorrectly uses line 16 ($7,709.50) as total tax without considering line 17 (0)"
          ],
          "evidence": [
            {
              "type": "input_field",
              "location": "input.return_data.irs1040.filing_status",
              "value": "single"
            }
          ]
        },
        {
          "form": "1040",
          "line": "37",
          "claim": "Amount you owe",
          "verdict": "wrong",
          "confidence": 0.8,
          "arguments": [
            "Draft incorrectly uses total tax ($7,709.50) instead of adjusted tax ($7,709.50 - $3,000 = $4,709.50)"
          ],
          "evidence": [
            {
              "type": "input_field",
              "location": "input.return_data.irs1040.filing_status",
              "value": "single"
            }
          ]
        }
      ]
    },
    "decision": {
      "decision": "needs_revision",
      "decision_confidence": 0.95,
      "comments": "Lines 9, 11, 16, 24, and 37 are incorrectly calculated. Total income should be $31,000 (30k from 1099-R + 1k dividend), AGI should be $31,000, tax calculation is incorrect, total tax should account for withheld payments, and amount owed is miscalculated.",
      "required_changes": {
        "lines_to_recompute": [
          {
            "form": "1040",
            "line": "9"
          },
          {
            "form": "1040",
            "line": "11"
          },
          {
            "form": "1040",
            "line": "16"
          },
          {
            "form": "1040",
            "line": "24"
          },
          {
            "form": "1040",
            "line": "37"
          }
        ]
      }
    }
  }
]
```