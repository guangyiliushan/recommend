"""ESMM — Entire Space Multi-task Model (Alibaba, SIGIR 2018).

Ma et al. "Entire Space Multi-Task Model: An Effective Approach
for Estimating Post-Click Conversion Rate"
- Solves Sample Selection Bias (SSB) and Data Sparsity (DS)
- CTR * CVR = CTCVR identity constraint
- Feature representation transfer from CTR tower to CVR tower
- No direct CVR labels needed — supervision from impression space
"""
# TODO: Implement ESMM(BaseRecommender)
#   task_type: "pcvr", @MODEL_REGISTRY.register("ESMM", family="pcvr", year=2018)
