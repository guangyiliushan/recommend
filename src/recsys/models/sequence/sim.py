"""SIM — Search-based Interest Model (2020).

Pi et al. "Search-based User Interest Modeling with Lifelong
Sequential Behavior Data for Click-Through Rate Prediction"
- Two-stage: General Search Unit (GSU) + Exact Search Unit (ESU)
- GSU: sub-linearly search for relevant behaviors from long history
- ESU: attention over selected behaviors
- Handles 54000-length behavior sequences
"""
# TODO: Implement SIM(BaseRecommender)
#   @MODEL_REGISTRY.register("SIM", family="sequence", year=2020)
