"""SASRec — Self-Attentive Sequential Recommendation (2018).

Kang & McAuley. "Self-Attentive Sequential Recommendation"
- Self-attention (Transformer encoder) for sequential recommendation
- Captures long-term semantics like RNN, but with fewer parameters
- Unidirectional attention mask (causal)
- Stochastic shared embedding dropout
"""
# TODO: Implement SASRec(BaseRecommender)
#   @MODEL_REGISTRY.register("SASRec", family="sequence", year=2018)
