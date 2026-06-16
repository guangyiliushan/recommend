"""DeepFM (Huawei, 2017).

Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
- End-to-end FM + DNN without any feature engineering
- FM component for low-order interactions, DNN for high-order
- Shared input embedding between FM and DNN
- No pre-training needed (unlike Wide&Deep)
"""
# TODO: Implement DeepFM(BaseRecommender)
#   @MODEL_REGISTRY.register("DeepFM", family="deep_ctr", year=2017)
