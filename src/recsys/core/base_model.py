"""BaseRecommender — 统一模型契约与能力接口。

设计原则：
- 统一外部契约，不统一内部实现
- 能力显式化，pipeline 基于能力路由
- 输出统一收敛到 PredictionBundle
- 最小稳定面，基类只定义所有模型共有的东西

结构：
- ModelOutput: 模型内部统一输出
- Batch: 受控的标准 batch 视图
- BaseRecommender: 所有模型共享的最薄父类
- 能力接口: Trainable, PointwisePredictor, Ranker, MultiTaskPredictor, Recommender, Checkpointable, Embeddable
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union, runtime_checkable

import torch
from torch import Tensor

from recsys.core.prediction_bundle import PredictionBundle


# ---------------------------------------------------------------------------
# 错误码定义
# ---------------------------------------------------------------------------

class ModelErrorCode(Enum):
    """模型层统一错误码。"""
    MODEL_CONTRACT_ERROR = "MODEL_CONTRACT_ERROR"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    MODEL_STATE_ERROR = "MODEL_STATE_ERROR"
    MODEL_SCHEMA_MISMATCH = "MODEL_SCHEMA_MISMATCH"


class ModelContractError(Exception):
    """模型契约错误。"""
    
    def __init__(
        self,
        message: str,
        code: ModelErrorCode = ModelErrorCode.MODEL_CONTRACT_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------------------
# Batch — 受控的标准 batch 视图
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """受控的标准 batch 视图。
    
    由 dataset 字典或 collate 结果转换而来，
    提供统一的字段访问和校验能力。
    
    字段分层与 datasets.md 对齐：
    - 标识字段: sample_id, user_id, item_id, group_id, session_id
    - 监督字段: label, labels, target_item_id
    - 候选字段: candidate_item_ids, candidate_mask
    - 历史字段: history_item_ids, history_mask
    - 特征字段: user_feats, item_feats, context_feats
    - 多任务字段: task_labels, task_masks
    - 多模态字段: text_emb, image_emb, video_emb
    
    Attributes
    ----------
    data : Dict[str, Any]
        原始 batch 数据字典。
    """
    
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 标识字段
    @property
    def sample_id(self) -> Optional[Tensor]:
        return self.data.get("sample_id")
    
    @property
    def user_id(self) -> Optional[Tensor]:
        return self.data.get("user_id")
    
    @property
    def item_id(self) -> Optional[Tensor]:
        return self.data.get("item_id")
    
    @property
    def group_id(self) -> Optional[Tensor]:
        return self.data.get("group_id")
    
    @property
    def session_id(self) -> Optional[Tensor]:
        return self.data.get("session_id")
    
    # 监督字段
    @property
    def label(self) -> Optional[Tensor]:
        return self.data.get("label")
    
    @property
    def labels(self) -> Optional[Tensor]:
        return self.data.get("labels")
    
    @property
    def target_item_id(self) -> Optional[Tensor]:
        return self.data.get("target_item_id")
    
    # 候选字段
    @property
    def candidate_item_ids(self) -> Optional[Tensor]:
        return self.data.get("candidate_item_ids")
    
    @property
    def candidate_mask(self) -> Optional[Tensor]:
        return self.data.get("candidate_mask")
    
    # 历史字段
    @property
    def history_item_ids(self) -> Optional[Tensor]:
        return self.data.get("history_item_ids")
    
    @property
    def history_mask(self) -> Optional[Tensor]:
        return self.data.get("history_mask")
    
    # 特征字段
    @property
    def user_feats(self) -> Optional[Tensor]:
        return self.data.get("user_feats")
    
    @property
    def item_feats(self) -> Optional[Tensor]:
        return self.data.get("item_feats")
    
    @property
    def context_feats(self) -> Optional[Tensor]:
        return self.data.get("context_feats")
    
    # 多任务字段
    @property
    def task_labels(self) -> Optional[Dict[str, Tensor]]:
        return self.data.get("task_labels")
    
    @property
    def task_masks(self) -> Optional[Dict[str, Tensor]]:
        return self.data.get("task_masks")
    
    # 多模态字段
    @property
    def text_emb(self) -> Optional[Tensor]:
        return self.data.get("text_emb")
    
    @property
    def image_emb(self) -> Optional[Tensor]:
        return self.data.get("image_emb")
    
    @property
    def video_emb(self) -> Optional[Tensor]:
        return self.data.get("video_emb")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取字段值。"""
        return self.data.get(key, default)
    
    def has(self, key: str) -> bool:
        """检查字段是否存在。"""
        return key in self.data
    
    def validate_required_features(self, required_features: List[str]) -> List[str]:
        """校验必需字段是否存在。
        
        Parameters
        ----------
        required_features : List[str]
            必需字段列表。
        
        Returns
        -------
        List[str]
            缺失字段列表，空列表表示全部存在。
        """
        missing = []
        for feat in required_features:
            if not self.has(feat):
                missing.append(feat)
        return missing
    
    @property
    def batch_size(self) -> int:
        """获取 batch 大小。"""
        for key in ["user_id", "item_id", "sample_id", "label"]:
            if self.has(key):
                tensor = self.get(key)
                if tensor is not None:
                    return tensor.shape[0]
        return 0


# ---------------------------------------------------------------------------
# ModelOutput — 模型内部统一输出
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """模型内部统一输出。
    
    它是模型前向或推理的标准中间输出，
    面向 trainer / inference adapter，不直接面向 evaluator。
    
    所有字段可选，但语义稳定：
    - trainer 主要消费 losses
    - inference adapter 主要消费 scores/probs/task_outputs
    - evaluator 不直接消费它，而是经 PredictionBundle 转换
    
    Attributes
    ----------
    scores : Tensor, optional
        模型输出的连续分数。
    logits : Tensor, optional
        模型输出的原始 logit。
    probs : Tensor, optional
        模型输出的概率（经过 sigmoid/softmax）。
    preds : Tensor, optional
        阈值化后的离散预测。
    embeddings : Tensor, optional
        输出的 embedding 向量。
    losses : Dict[str, Tensor], optional
        各损失项字典。
    aux_outputs : Dict[str, Any], optional
        辅助输出字典。
    task_outputs : Dict[str, Tensor], optional
        多任务各任务头的输出。
    attentions : Dict[str, Tensor], optional
        注意力权重字典。
    metadata : Dict[str, Any], optional
        附加元信息。
    """
    
    scores: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    probs: Optional[Tensor] = None
    preds: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None
    aux_outputs: Optional[Dict[str, Any]] = None
    task_outputs: Optional[Dict[str, Tensor]] = None
    attentions: Optional[Dict[str, Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_loss(self) -> bool:
        """是否有损失。"""
        return self.losses is not None and len(self.losses) > 0
    
    @property
    def total_loss(self) -> Optional[Tensor]:
        """总损失。"""
        if not self.has_loss:
            return None
        return sum(self.losses.values())


# ---------------------------------------------------------------------------
# 能力定义
# ---------------------------------------------------------------------------

class Capability(Enum):
    """模型能力枚举。"""
    TRAINABLE = auto()           # 可训练
    POINTWISE_PREDICTOR = auto() # 点预测
    RANKER = auto()              # 排序
    MULTITASK = auto()           # 多任务
    RECOMMENDER = auto()         # 推荐
    CHECKPOINTABLE = auto()      # 可持久化
    EMBEDDABLE = auto()          # 可导出 embedding


# ---------------------------------------------------------------------------
# 能力接口（Protocol）
# ---------------------------------------------------------------------------

@runtime_checkable
class Trainable(Protocol):
    """可训练能力接口。
    
    适用于 DeepFM、DIN、Transformer 类可训练模型。
    """
    
    def forward(self, batch: Batch) -> ModelOutput:
        """前向传播。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        
        Returns
        -------
        ModelOutput
            模型输出。
        """
        ...
    
    def compute_loss(
        self, 
        batch: Batch, 
        output: ModelOutput
    ) -> Dict[str, Tensor]:
        """计算损失。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        output : ModelOutput
            模型输出。
        
        Returns
        -------
        Dict[str, Tensor]
            损失字典。
        """
        ...


@runtime_checkable
class PointwisePredictor(Protocol):
    """点预测能力接口。
    
    定义样本级打分/概率输出，面向 CTR/CVR/评分预测。
    """
    
    def predict(self, batch: Batch) -> ModelOutput:
        """预测。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        
        Returns
        -------
        ModelOutput
            模型输出，应包含 scores 或 probs。
        """
        ...


@runtime_checkable
class Ranker(Protocol):
    """排序能力接口。
    
    定义组内候选打分，面向 retrieval/ranking/listwise。
    """
    
    def rank(
        self, 
        batch: Batch, 
        candidate_ids: Optional[Tensor] = None
    ) -> ModelOutput:
        """排序打分。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        candidate_ids : Tensor, optional
            候选物品 ID。
        
        Returns
        -------
        ModelOutput
            模型输出，应包含每个候选的分数。
        """
        ...


@runtime_checkable
class MultiTaskPredictor(Protocol):
    """多任务预测能力接口。
    
    定义多任务头输出，面向 ESMM/ESM2/漏斗建模。
    """
    
    def predict_multitask(self, batch: Batch) -> ModelOutput:
        """多任务预测。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        
        Returns
        -------
        ModelOutput
            模型输出，应包含 task_outputs。
        """
        ...


@runtime_checkable
class Recommender(Protocol):
    """推荐能力接口。
    
    定义 recommend(user_ids, top_k, candidates=None) 这类面向在线/离线推荐接口。
    """
    
    def recommend(
        self,
        user_ids: Tensor,
        top_k: int,
        candidates: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """推荐。
        
        Parameters
        ----------
        user_ids : Tensor
            用户 ID 张量。
        top_k : int
            推荐数量。
        candidates : Tensor, optional
            候选物品 ID。
        
        Returns
        -------
        Tuple[Tensor, Tensor]
            (推荐物品 ID, 分数)
        """
        ...


@runtime_checkable
class Checkpointable(Protocol):
    """可持久化能力接口。
    
    仅可训练且需要持久化的模型实现。
    """
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点。
        
        Parameters
        ----------
        path : str
            保存路径。
        """
        ...
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点。
        
        Parameters
        ----------
        path : str
            检查点路径。
        """
        ...


@runtime_checkable
class Embeddable(Protocol):
    """可导出 embedding 能力接口。
    
    为双塔、序列、生成式检索提供 embedding 导出能力。
    """
    
    def encode_user(self, batch_or_ids: Union[Batch, Tensor]) -> Tensor:
        """编码用户。
        
        Parameters
        ----------
        batch_or_ids : Union[Batch, Tensor]
            Batch 或用户 ID。
        
        Returns
        -------
        Tensor
            用户 embedding。
        """
        ...
    
    def encode_item(self, batch_or_ids: Union[Batch, Tensor]) -> Tensor:
        """编码物品。
        
        Parameters
        ----------
        batch_or_ids : Union[Batch, Tensor]
            Batch 或物品 ID。
        
        Returns
        -------
        Tensor
            物品 embedding。
        """
        ...


# ---------------------------------------------------------------------------
# BaseRecommender — 所有模型共享的最薄父类
# ---------------------------------------------------------------------------

class BaseRecommender(ABC):
    """所有推荐模型共享的最薄父类。
    
    设计原则：
    - 统一外部契约，不统一内部实现
    - 能力显式化，pipeline 基于能力路由
    - 输出统一收敛到 PredictionBundle
    - 最小稳定面，基类只定义所有模型共有的东西
    
    不强制要求所有模型实现训练 forward/loss，
    只负责元信息、状态、公共校验、预测产物导出入口。
    
    Attributes
    ----------
    model_name : str
        模型名称。
    model_family : str
        模型家族。
    task_type : str
        任务类型：pointwise / ranking / multitask。
    problem_type : str
        问题类型：binary / multiclass / multilabel / regression / implicit_ranking / listwise_ranking。
    supports_training : bool
        是否支持训练。
    required_features : List[str]
        必需的输入字段。
    default_metrics : List[str]
        默认评估指标。
    """
    
    # 类级别元信息（子类应覆盖）
    model_name: str = "base"
    model_family: str = "base"
    task_type: str = "pointwise"
    problem_type: str = "binary"
    supports_training: bool = False
    required_features: List[str] = []
    default_metrics: List[str] = []
    
    def __init__(
        self,
        config: Optional[Any] = None,
        schema_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化模型。
        
        Parameters
        ----------
        config : Any, optional
            模型配置。
        schema_metadata : Dict[str, Any], optional
            数据 schema 元信息。
        **kwargs : Any
            其他参数。
        """
        self._config = config
        self._schema_metadata = schema_metadata or {}
        self._kwargs = kwargs
        
        # 生命周期状态
        self._initialized: bool = True
        self._fitted: bool = False
        self._trained: bool = False
        self._device: str = "cpu"
        self._schema_version: str = "1.0.0"
        
        # 能力集合
        self._capabilities: Set[Capability] = set()
        self._init_capabilities()
    
    def _init_capabilities(self) -> None:
        """初始化能力集合。子类可覆盖。"""
        if self.supports_training:
            self._capabilities.add(Capability.TRAINABLE)
    
    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    
    @classmethod
    def get_model_metadata(cls) -> Dict[str, Any]:
        """获取模型静态元信息。
        
        Returns
        -------
        Dict[str, Any]
            模型元信息字典。
        """
        return {
            "name": cls.model_name,
            "family": cls.model_family,
            "task_type": cls.task_type,
            "problem_type": cls.problem_type,
            "supports_training": cls.supports_training,
            "required_features": cls.required_features,
            "default_metrics": cls.default_metrics,
        }
    
    def get_runtime_capabilities(self) -> Set[Capability]:
        """获取运行时能力集合。
        
        Returns
        -------
        Set[Capability]
            能力集合。
        """
        return self._capabilities.copy()
    
    def supports(self, capability: Union[Capability, str]) -> bool:
        """检查是否支持某能力。
        
        Parameters
        ----------
        capability : Union[Capability, str]
            能力枚举或名称。
        
        Returns
        -------
        bool
            是否支持。
        """
        if isinstance(capability, str):
            try:
                capability = Capability[capability.upper()]
            except KeyError:
                return False
        return capability in self._capabilities
    
    # ------------------------------------------------------------------
    # 生命周期状态
    # ------------------------------------------------------------------
    
    @property
    def initialized(self) -> bool:
        """是否已初始化。"""
        return self._initialized
    
    @property
    def fitted(self) -> bool:
        """是否已拟合（非训练式模型）。"""
        return self._fitted
    
    @property
    def trained(self) -> bool:
        """是否已训练（训练式模型）。"""
        return self._trained
    
    @property
    def device(self) -> str:
        """当前设备。"""
        return self._device
    
    @device.setter
    def device(self, value: str) -> None:
        self._device = value
    
    @property
    def schema_version(self) -> str:
        """schema 版本。"""
        return self._schema_version
    
    # ------------------------------------------------------------------
    # 公共校验
    # ------------------------------------------------------------------
    
    def validate_batch(self, batch: Batch) -> None:
        """校验 batch 是否满足模型要求。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        
        Raises
        ------
        ModelContractError
            缺少必需字段时抛出。
        """
        missing = batch.validate_required_features(self.required_features)
        if missing:
            raise ModelContractError(
                f"Batch 缺少必需字段: {missing}",
                code=ModelErrorCode.MODEL_SCHEMA_MISMATCH,
                details={"missing_features": missing},
            )
    
    def validate_prediction_bundle(self, bundle: PredictionBundle) -> None:
        """校验 PredictionBundle 是否有效。
        
        Parameters
        ----------
        bundle : PredictionBundle
            预测产物。
        
        Raises
        ------
        ModelContractError
            bundle 校验失败时抛出。
        """
        errors = bundle.validate()
        if errors:
            raise ModelContractError(
                f"PredictionBundle 校验失败: {'; '.join(errors)}",
                code=ModelErrorCode.MODEL_CONTRACT_ERROR,
                details={"validation_errors": errors},
            )
    
    # ------------------------------------------------------------------
    # 参数可观测性
    # ------------------------------------------------------------------
    
    def count_parameters(self) -> int:
        """统计模型参数数量。
        
        Returns
        -------
        int
            参数总数。
        """
        # 默认实现：非神经网络模型返回 0
        return 0
    
    def trainable_parameters(self) -> int:
        """统计可训练参数数量。
        
        Returns
        -------
        int
            可训练参数数量。
        """
        # 默认实现：非神经网络模型返回 0
        return 0
    
    def extra_repr(self) -> str:
        """额外的字符串表示。
        
        Returns
        -------
        str
            额外信息字符串。
        """
        return ""
    
    def __repr__(self) -> str:
        """字符串表示。"""
        extra = self.extra_repr()
        if extra:
            return f"{self.__class__.__name__}({extra})"
        return f"{self.__class__.__name__}()"
    
    # ------------------------------------------------------------------
    # 核心方法（子类应实现）
    # ------------------------------------------------------------------
    
    def fit(self, *args: Any, **kwargs: Any) -> "BaseRecommender":
        """拟合模型（非训练式模型）。
        
        Parameters
        ----------
        *args : Any
            位置参数。
        **kwargs : Any
            关键字参数。
        
        Returns
        -------
        BaseRecommender
            self。
        
        Raises
        ------
        ModelContractError
            模型不支持拟合时抛出。
        """
        if not self.supports(Capability.TRAINABLE):
            raise ModelContractError(
                f"模型 {self.model_name} 不支持 fit",
                code=ModelErrorCode.UNSUPPORTED_CAPABILITY,
            )
        # 子类应覆盖此方法
        self._fitted = True
        return self
    
    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> PredictionBundle:
        """预测并返回 PredictionBundle。
        
        所有模型必须实现此方法，输出统一收敛到 PredictionBundle。
        
        Parameters
        ----------
        *args : Any
            位置参数。
        **kwargs : Any
            关键字参数。
        
        Returns
        -------
        PredictionBundle
            统一预测产物。
        """
        ...
    
    def export_prediction_bundle(
        self,
        y_true: List[Any],
        y_score: List[Any],
        **kwargs: Any,
    ) -> PredictionBundle:
        """导出 PredictionBundle。
        
        提供便捷方法，帮助模型构造标准 PredictionBundle。
        
        Parameters
        ----------
        y_true : List[Any]
            真实标签。
        y_score : List[Any]
            预测分数。
        **kwargs : Any
            其他字段。
        
        Returns
        -------
        PredictionBundle
            统一预测产物。
        """
        bundle = PredictionBundle(
            task_type=self.task_type,
            problem_type=self.problem_type,
            y_true=y_true,
            y_score=y_score,
            metadata={
                "model_name": self.model_name,
                "model_family": self.model_family,
                **kwargs.pop("metadata", {}),
            },
            **kwargs,
        )
        self.validate_prediction_bundle(bundle)
        return bundle


# ---------------------------------------------------------------------------
# 神经网络模型基类
# ---------------------------------------------------------------------------

class NeuralRecommender(BaseRecommender, torch.nn.Module):
    """神经网络推荐模型基类。
    
    继承 BaseRecommender 和 torch.nn.Module，
    提供神经网络模型的基本能力。
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        schema_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化神经网络模型。
        
        Parameters
        ----------
        config : Any, optional
            模型配置。
        schema_metadata : Dict[str, Any], optional
            数据 schema 元信息。
        **kwargs : Any
            其他参数。
        """
        BaseRecommender.__init__(self, config, schema_metadata, **kwargs)
        torch.nn.Module.__init__(self)
        
        # 神经网络模型默认支持训练
        self._capabilities.add(Capability.TRAINABLE)
    
    def count_parameters(self) -> int:
        """统计模型参数数量。"""
        return sum(p.numel() for p in self.parameters())
    
    def trainable_parameters(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, batch: Batch) -> ModelOutput:
        """前向传播。
        
        子类应实现此方法。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        
        Returns
        -------
        ModelOutput
            模型输出。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 forward(batch) -> ModelOutput"
        )
    
    def compute_loss(
        self,
        batch: Batch,
        output: ModelOutput,
    ) -> Dict[str, Tensor]:
        """计算损失。
        
        子类应实现此方法。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        output : ModelOutput
            模型输出。
        
        Returns
        -------
        Dict[str, Tensor]
            损失字典。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 compute_loss(batch, output) -> Dict[str, Tensor]"
        )
    
    def predict(self, batch: Batch, **kwargs: Any) -> PredictionBundle:
        """预测并返回 PredictionBundle。
        
        默认实现：调用 forward 并转换为 PredictionBundle。
        子类可覆盖以提供更复杂的逻辑。
        
        Parameters
        ----------
        batch : Batch
            标准 batch 视图。
        **kwargs : Any
            其他参数。
        
        Returns
        -------
        PredictionBundle
            统一预测产物。
        """
        self.validate_batch(batch)
        
        output = self.forward(batch)
        
        y_true = batch.label.cpu().tolist() if batch.label is not None else []
        y_score = (
            output.probs.cpu().tolist() 
            if output.probs is not None 
            else (output.scores.cpu().tolist() if output.scores is not None else [])
        )
        
        return self.export_prediction_bundle(
            y_true=y_true,
            y_score=y_score,
            **kwargs,
        )
