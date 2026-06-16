"""BERT4Rec — Bidirectional Encoder for Sequential Recommendation (2019).

Sun et al. "BERT4Rec: Sequential Recommendation with
Bidirectional Encoder Representations from Transformer"
- Cloze task (Masked Item Prediction) instead of left-to-right
- Bidirectional self-attention without causal masking
- More powerful than SASRec (unidirectional)
"""
# TODO: Implement BERT4Rec(BaseRecommender)
#   @MODEL_REGISTRY.register("BERT4Rec", family="sequence", year=2019)
