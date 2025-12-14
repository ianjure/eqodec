"""
Example: Computing EQODEC Energy-Efficiency Score (EES).

This works for:
- Neural codecs
- Traditional codecs
- Any compression method with size + carbon cost
"""

from eqodec import energy_efficiency_score

# ----------------------------------------
# Example compression results (hypothetical)
# ----------------------------------------

# Baseline (e.g., original H.264 or raw storage)
baseline_bytes = 2.45 * 1024**3   # 2.45 GB

# Compressed output (neural codec, optimized model, etc.)
compressed_bytes = 2.05 * 1024**3  # 2.05 GB

# Measured carbon cost of inference + encoding
kg_co2_overhead = 0.18  # kgCO2


# ----------------------------------------
# Compute Energy-Efficiency Score
# ----------------------------------------
ees = energy_efficiency_score(
    baseline_bytes=baseline_bytes,
    compressed_bytes=compressed_bytes,
    kg_co2=kg_co2_overhead
)

print("Energy-Efficiency Score (EES)")
print(f"{ees:.4f} GB saved per kgCO2")
