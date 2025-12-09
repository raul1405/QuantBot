"""
LLM Council: The Skeptical Review Layer
=======================================

This tool implements the "Council Review" workflow. 
It aggregates context (diffs, metrics, configs) and generates specific 
prompts for 5 distinct Risk/Quant personas.

Usage:
    python tools/council.py --action FREEZE_SPEC --context "notes/frozen_v3.md"
    python tools/council.py --action DECLARE_PASS --context "notes/strategy_checklist.md"

Design:
    - Reads artifacts from filesystem.
    - Generates 5 distinct prompts (one per role).
    - (Future) Can connect to LLM API to get automated verdicts.
    - Currently: Generates the "Review Packet" for the Operator (Agent) to evaluate.
"""

import argparse
import sys
import os
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

COUNCIL_ROLES = {
    "QuantResearcher": {
        "persona": "You are a cynical, senior quantitative researcher with 20 years of experience. You Assume every backtest is overfitting.",
        "focus": ["Statistical Significance", "Train/Test Separation", "p-hacking"]
    },
    "RiskOfficer": {
        "persona": "You are a strict Risk Manager ensuring compliance with FTMO prop firm rules. You care about survival, not profit.",
        "focus": ["Max Drawdown", "Leverage Caps", "Weekend Risk", "News Events"]
    },
    "CodeAuditor": {
        "persona": "You are a senior software engineer. Look only at code logic, bugs, race conditions, and implementation gaps.",
        "focus": ["Logic Bugs", "Look-ahead Bias", "Variable Usage", "Error Handling"]
    },
    "RealismChecker": {
        "persona": "You are an ex-floor trader. You care about slippage, spread, liquidity, and execution reality.",
        "focus": ["Transaction Costs", "Liquidity", "Data Quality", "Microstructure"]
    },
    "IntegrationEngineer": {
        "persona": "You are the DevOps Architect. You care about system stability, logging, dependencies, and backward compatibility.",
        "focus": ["Dependencies", "Config Schema", "Logging", "Deployment Manifest"]
    },
    "OverfittingGuard": {
        "persona": "You are a specialized statistician focused solely on 'P-Hacking' and 'Look-Ahead Bias'. You trust NOTHING.",
        "focus": ["Train/Test Leakage", "Parameter Tuning Abuse", "Selection Bias"]
    }
}

# ==============================================================================
# UTILITIES
# ==============================================================================

def get_git_diff() -> str:
    """Get the current staged/unstaged diff."""
    try:
        # Diff of staged + unstaged changes
        res = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
        return res.stdout[:10000] # Cap at 10k chars to fit context
    except Exception as e:
        return f"Error getting git diff: {e}"

def read_file(path: str) -> str:
    """Read a file safely."""
    try:
        if not os.path.exists(path):
            return f"[MISSING FILE: {path}]"
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"[ERROR READING {path}: {e}]"

# ==============================================================================
# COUNCIL ENGINE
# ==============================================================================

class CouncilEngine:
    def __init__(self):
        self.evidence = {}
        
    def gather_evidence(self, action: str, context_files: List[str]):
        """Collects all necessary artifacts for the review."""
        print(f"🕵️  Council Gathering Evidence for acting: {action}...")
        
        # 1. Git Diff (Code changes)
        self.evidence['git_diff'] = get_git_diff()
        
        # 2. Context Files (The proposal)
        self.evidence['files'] = {}
        for fpath in context_files:
            self.evidence['files'][fpath] = read_file(fpath)
            
        # 3. Strategy Checklist (Standard Artifact)
        self.evidence['checklist'] = read_file("notes/strategy_evaluation_checklist.md")
        
        # 4. Config (Standard Artifact)
        # Try to read live_config.json or setup_config.py
        if os.path.exists("live_config.json"):
            self.evidence['config'] = read_file("live_config.json")
        else:
            self.evidence['config'] = read_file("setup_config.py")

    def generate_review_packet(self) -> str:
        """Generates the full prompt packet for the Agent/User to evaluate."""
        
        packet = []
        packet.append("# 🛡️ LLM COUNCIL REVIEW PACKET")
        packet.append(f"**Date:** {datetime.now().isoformat()}")
        packet.append("")
        
        packet.append("## 📂 EVIDENCE SUBMITTED")
        packet.append(f"- **Git Diff Size:** {len(self.evidence.get('git_diff', ''))} chars")
        packet.append(f"- **Files Submitted:** {list(self.evidence['files'].keys())}")
        packet.append("")
        
        # For each role, create a specific prompt section
        for role, info in COUNCIL_ROLES.items():
            packet.append(f"---")
            packet.append(f"## 👤 ROLE: {role}")
            packet.append(f"**PERSONA:** {info['persona']}")
            packet.append(f"**FOCUS:** {', '.join(info['focus'])}")
            packet.append("")
            packet.append("**PROTOCOL: STRICT INDEPENDENCE**")
            packet.append("You are in a Sound-Proof Room. You have NO KNOWLEDGE of the other Council Members' opinions. Judge solely on the Evidence.")
            packet.append("")
            packet.append("### INSTRUCTIONS")
            packet.append("Review the Evidence below and provide a verdict JSON:")
            packet.append("```json")
            packet.append("{")
            packet.append('  "role": "' + role + '",')
            packet.append('  "verdict": "APPROVE" | "REJECT" | "NEEDS_MORE_INFO",')
            packet.append('  "confidence": 0.0 to 1.0,')
            packet.append('  "concerns": ["..."],')
            packet.append('  "required_actions": ["..."]')
            packet.append("}")
            packet.append("```")
            packet.append("")
        
        packet.append("---")
        packet.append("## 📄 EVIDENCE DUMP")
        
        packet.append("\n### [GIT DIFF HEAD]")
        packet.append("```diff")
        packet.append(self.evidence.get('git_diff', 'No Diff'))
        packet.append("```")
        
        for fname, content in self.evidence['files'].items():
            packet.append(f"\n### [FILE: {fname}]")
            packet.append("```")
            packet.append(content[:5000]) # Cap file content
            if len(content) > 5000: packet.append("\n... (truncated)")
            packet.append("```")

        return "\n".join(packet)

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Council Review")
    parser.add_argument("--action", type=str, required=True, 
                        help="Action type (FREEZE_SPEC, UPDATE_CONFIG, DECLARE_PASS, CODE_REFACTOR)")
    parser.add_argument("--context", type=str, nargs='+', required=True, 
                        help="List of file paths relevant to the review")
    
    args = parser.parse_args()
    
    engine = CouncilEngine()
    engine.gather_evidence(args.action, args.context)
    packet = engine.generate_review_packet()
    
    # Write packet to a file for the agent to read
    out_path = "notes/council_review_packet.md"
    with open(out_path, "w") as f:
        f.write(packet)
        
    print(f"\n✅ REVIEW PACKET GENERATED: {out_path}")
    print("The Agent must now read this file and simulate the 5 Council Roles.")
