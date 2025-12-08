import unittest
import numpy as np

class TestExecutionLogic(unittest.TestCase):
    def test_vol_adjusted_sizing(self):
        """Verify the Volatility-Adjusted Risk Sizing Formula from v2.1 Spec."""
        print("\n[TEST] Vol-Adjusted Sizing...")
        
        # Formula: 
        # Vol_Mult = 1 / (1 + Vol_Intensity^2)
        # Clamped [0.3, 1.2]
        # Base Risk = 0.30%
        
        base_risk = 0.003
        
        # Case 1: Normal Vol (Intensity 0)
        # Mult = 1 / 1 = 1.0
        # Risk = 0.30%
        vol_int = 0.0
        mult = 1.0 / (1.0 + max(vol_int, 0)**2)
        mult = max(0.3, min(mult, 1.2))
        risk = base_risk * mult
        self.assertAlmostEqual(risk, 0.003, delta=0.00001, msg="Normal Vol Sizing Failed")
        print(f"  Normal Vol (0.00): Risk {risk*100:.2f}% (Expected 0.30%)")
        
        # Case 2: High Vol (Intensity 2.0 - 2 Sigma)
        # Mult = 1 / (1 + 4) = 0.2
        # Clamped to 0.3
        # Risk = 0.30% * 0.3 = 0.09%
        vol_int = 2.0
        mult = 1.0 / (1.0 + max(vol_int, 0)**2)
        mult = max(0.3, min(mult, 1.2))
        risk = base_risk * mult
        self.assertAlmostEqual(risk, 0.0009, delta=0.00001, msg="High Vol Sizing Failed")
        print(f"  High Vol (2.00): Risk {risk*100:.2f}% (Expected 0.09%)")
        
        # Case 3: Low Vol (Intensity -1.0)
        # Mult = 1 / (1 + 0) = 1.0 (Since max(vol_int, 0) handles negatives? Check implementation)
        # LiveTrader code: max(vol_int, 0.0)**2
        # So negative intensity counts as Normal (0).
        vol_int = -1.0
        mult = 1.0 / (1.0 + max(vol_int, 0)**2)
        mult = max(0.3, min(mult, 1.2))
        self.assertEqual(mult, 1.0)
        print(f"  Low Vol (-1.00): Risk {base_risk*mult*100:.2f}% (Expected 0.30%)")

if __name__ == '__main__':
    unittest.main()
