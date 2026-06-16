"""BaseRecommender — abstract base class for all recommendation models.

All 50+ algorithms must implement:
    forward(batch: Batch) -> ModelOutput
    compute_loss(batch, output) -> Dict[str, Tensor]

Optional overrides:
    predict(batch) -> ModelOutput
    recommend(user_ids, top_k) -> Tensor

Metadata classmethods:
    model_name() / model_family() / supported_tasks() / required_features()
"""
# TODO: Implement:
#   - ModelOutput dataclass (logits, probs, embeddings, aux_outputs, loss)
#   - Batch dataclass (features, labels, aux_labels, mask)
#   - BaseRecommender(ABC, nn.Module) with abstract forward/compute_loss
#   - predict/recommend default implementations
#   - count_parameters() / estimate_flops()
