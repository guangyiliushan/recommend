"""YouTube DNN (Covington et al., 2016).

Covington, Adams, Sargin. "Deep Neural Networks for YouTube Recommendations"
- Two-stage: candidate generation (recall) + ranking
- Candidate generation: classify next watch from corpus via softmax
- Ranking: weighted logistic regression with expected watch time
- Established the recall+ranking paradigm
"""
# TODO: Implement YouTubeDNN(BaseRecommender)
#   @MODEL_REGISTRY.register("YouTubeDNN", family="deep_ctr", year=2016)
