"""DCNv2 — Deep & Cross Network V2 (Google, 2021).

Wang et al. "DCN V2: Improved Deep & Cross Network and Practical
Lessons for Web-scale Learning"
- Cross network v2: matrix W instead of vector w → expressiveness ↑
- Mixture of low-rank: decomposes W = U V^T to save parameters
- Stacked + parallel architectures
"""
# TODO: Implement DCNv2(BaseRecommender)
#   @MODEL_REGISTRY.register("DCNv2", family="feature_cross", year=2021)
