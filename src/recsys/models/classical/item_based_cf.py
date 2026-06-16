"""Item-based Collaborative Filtering (Amazon, 2001).

Sarwar et al. "Item-based collaborative filtering recommendation algorithms"
- Builds item-item similarity matrix from user-item interactions
- Cosine similarity with Top-K truncation
- Predicts rating as weighted average of similar items
"""
# TODO: Implement ItemBasedCF(BaseRecommender)
#   family = "classical", task = ["ctr", "ranking"]
#   @MODEL_REGISTRY.register("ItemCF", family="classical", year=2001)
