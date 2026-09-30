from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal


class PlanTask(BaseModel):
    task_id: str = Field(..., description="Unique id for the task.")
    task_name: str = Field(..., description="Human-readable name.")
    task_goal: str = Field(..., description="What success looks like for the task.")
    schema_id: str = Field(..., description="Which output schema to return.")
    section_focus: List[str] = Field(default_factory=list, description="Section hints to focus on.")
    tool_hints: List[str] = Field(default_factory=list, description="Tool usage hints (e.g., retrieve_passages).")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Checklist for evaluator.")
    instructions: str = Field(..., description="Precise instructions to execute the task.")


class PlanTasks(BaseModel):
    tasks: List[PlanTask]


class SummaryOut(BaseModel):
    title: Optional[str] = None
    abstract_summary: str
    overall_summary: str
    contributions: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class Finding(BaseModel):
    claim: str
    evidence: str = Field(..., description="Grounded evidence (from paper passages).")
    section: Optional[str] = None


class FindingsOut(BaseModel):
    key_findings: List[Finding] = Field(default_factory=list)


class ResultsOut(BaseModel):
    results_summary: str
    metrics: List[str] = Field(default_factory=list)
    ablations: List[str] = Field(default_factory=list)


class LimitationsOut(BaseModel):
    limitations: List[str] = Field(default_factory=list)
    threats_to_validity: List[str] = Field(default_factory=list)


class ExperimentalSetupOut(BaseModel):
    datasets: List[str] = Field(default_factory=list)
    evaluation_protocol: str
    metrics: List[str] = Field(default_factory=list)
    baselines: List[str] = Field(default_factory=list)
    implementation_details: List[str] = Field(default_factory=list)


class ReplicationChecklistOut(BaseModel):
    prerequisites: List[str] = Field(default_factory=list)
    step_by_step: List[str] = Field(default_factory=list)
    hyperparameters: List[str] = Field(default_factory=list)
    gotchas: List[str] = Field(default_factory=list)


class PeerReviewOut(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    questions_for_authors: List[str] = Field(default_factory=list)
    overall_assessment: str


class AlgorithmExtractionOut(BaseModel):
    algorithm_name: Optional[str] = None
    high_level_idea: str
    pseudo_steps: List[str] = Field(default_factory=list)
    inputs_outputs: List[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    issue: str
    suggestion: str


class ReviewOut(BaseModel):
    grounded: bool = Field(..., description="True if key claims are supported by paper evidence.")
    issues: List[ReviewIssue] = Field(default_factory=list)


class FinalReport(BaseModel):
    outputs: Dict[str, dict] = Field(default_factory=dict)
    review: ReviewOut
    metadata: Dict[str, str] = Field(default_factory=dict)