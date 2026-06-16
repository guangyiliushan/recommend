"""ESM² — Entire Space Multi-task Model v2 (Alibaba, SIGIR 2020).

Wen et al. "Entire Space Multi-task Model v2: Beyond Click"
- Decomposes post-click behaviors into DAction and OAction paths
- Parallel modeling of two intermediate behaviors
- Shared embedding + separate DNN towers
- CTCVR = CTR * (DAction + OAction), no direct CVR supervision
"""
# TODO: Implement ESM2(BaseRecommender)
#   task_type: "pcvr", @MODEL_REGISTRY.register("ESM2", family="pcvr", year=2020)
