# Council API Contract

This document defines the interface for communicating with the Council.

The system communicates via JSON files dropped into a specific directory or passed as context strings.

## 1. Request Object (`CouncilRequest`)

This is the input provided by the Research Agent.

```json
{
  "request_id": "req_123456789",
  "timestamp": "2025-12-09T12:00:00Z",
  "proposer": "QuantResearchAgent_v1",
  "change_type": "DECLARE_PASS" | "FREEZE_SPEC" | "UPDATE_CONFIG" | "CODE_REFACTOR",
  
  "context": {
    "summary": "Declaring that Strategy v2.1 has passed validation based on new OOS tests.",
    "files_changed": ["notes/strategy_evaluation_checklist.md"],
    "key_metrics": {
       "total_return": 15.4,
       "sharpe": 1.2,
       "max_dd": -4.5,
       "oos_win_rate": 55.0
    }
  },

  "artifacts": {
     "spec_content": "... markdown string ...",
     "diff_content": "... git diff string ...",
     "validation_report": "... markdown table string ...",
     "monte_carlo_stats": "... json string ..."
  }
}
```

## 2. Role Answer Object (`RoleVerdict`)

This is the output from *each individual role* (e.g., Risk Officer).

```json
{
  "role": "Risk Officer",
  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",
  "confidence_score": 0.9,
  
  "checklist_results": [
    {"question": "Does it hold over weekend?", "answer": "No", "pass": true},
    {"question": "Max DD < limit?", "answer": "Yes, 4.5% vs 10%", "pass": true}
  ],
  
  "concerns": [
    "Leverage seems high for this volatility regime."
  ],
  
  "required_actions": [
    "Run stress test with 2x volatility."
  ]
}
```

## 3. Final Decision Object (`CouncilDecision`)

This is the aggregated output from the Council Chairperson.

```json
{
  "request_id": "req_123456789",
  "final_outcome": "REJECT",
  
  "axis_scores": {
    "assumptions": 8.0,
    "applicability": 10.0,
    "code_correctness": 9.0,
    "realistic_newness": 4.0, 
    "overfitting": 2.0,
    "implications": 7.0
  },
  
  "summary_reasoning": "The council REJECTS this proposal. While the code is correct and applicable, the Quant Researcher and Realism Checker flagged significant overfitting risks (Axis 5 score: 2.0). The reported Sharpe of 5.0 is statistically implausible.",
  
  "blocking_issues": [
    "In-Sample data leakage detected by Quant Researcher.",
    "OOS win rate is only 39%."
  ],
  
  "next_steps": [
    "Refactor validation script to enforce strict Train/Test split.",
    "Re-run backtest."
  ]
}
```
