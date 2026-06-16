"""DIN — Deep Interest Network (Alibaba, 2018).

Zhou et al. "Deep Interest Network for Click-Through Rate Prediction"
- Target Attention: compute user representation w.r.t. candidate item
- Activation Unit: attention between candidate and historical behaviors
- Dice activation: data-adaptive PReLU variant
- Mini-batch aware regularization
"""
# TODO: Implement DIN(BaseRecommender)
#   @MODEL_REGISTRY.register("DIN", family="deep_ctr", year=2018)
