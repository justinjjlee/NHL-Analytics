"""
models.py — Data structures and domain models for the Scoreboard Agent Framework.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class SkaterStats:
    """Individual skater performance metrics for a single game."""
    player_id: int
    name: str
    sweater_number: int
    position: str
    team_abbrev: str
    goals: int = 0
    assists: int = 0
    points: int = 0
    plus_minus: int = 0
    pim: int = 0
    sog: int = 0
    hits: int = 0
    blocked_shots: int = 0
    giveaways: int = 0
    takeaways: int = 0
    power_play_goals: int = 0
    faceoff_pct: float = 0.0
    toi: str = "00:00"
    shifts: int = 0
    ixg: float = 0.0
    hd_shots: int = 0

    @property
    def is_multi_goal(self) -> bool:
        return self.goals >= 2

    @property
    def is_hat_trick(self) -> bool:
        return self.goals >= 3

    @property
    def is_multi_point(self) -> bool:
        return self.points >= 2


@dataclass
class GoalieStats:
    """Individual goaltender performance metrics for a single game."""
    player_id: int
    name: str
    sweater_number: int
    team_abbrev: str
    saves: int = 0
    shots_against: int = 0
    goals_against: int = 0
    save_pct: float = 0.0
    toi: str = "60:00"
    decision: Optional[str] = None
    starter: bool = True
    even_strength_shots: str = "0/0"
    power_play_shots: str = "0/0"
    shorthanded_shots: str = "0/0"
    xga: float = 0.0
    gsax: float = 0.0

    @property
    def is_shutout(self) -> bool:
        return self.goals_against == 0 and self.shots_against > 0 and self.decision == "W"

    @property
    def is_brick_wall(self) -> bool:
        """35+ saves with >= .925 save percentage."""
        return self.saves >= 35 and self.save_pct >= 0.925


@dataclass
class TeamGameStats:
    """Team-level summary statistics for a single game."""
    team_id: int
    team_name: str
    team_abbrev: str
    score: int = 0
    sog: int = 0
    faceoff_pct: float = 0.0
    faceoff_wins: str = "0/0"
    power_play_conv: str = "0/0"
    power_play_pct: float = 0.0
    pim: int = 0
    hits: int = 0
    blocked_shots: int = 0
    giveaways: int = 0
    takeaways: int = 0
    logo_url: str = ""
    xg: float = 0.0
    xg_pct: float = 50.0
    hdcf: int = 0


@dataclass
class ScoringPlay:
    """Individual scoring event within a game."""
    period: int
    period_descriptor: str
    time_in_period: str
    scoring_team: str
    goal_scorer: str
    goal_scorer_id: int
    scorer_total_goals: int = 0
    assists: List[str] = field(default_factory=list)
    strength: str = "EV"  # EV, PP, SH, EN
    away_score: int = 0
    home_score: int = 0
    highlight_url: str = ""


@dataclass
class ThreeStar:
    """Star of the game designation."""
    star: int
    player_id: int
    name: str
    team_abbrev: str
    position: str
    stat_summary: str = ""
    headshot_url: str = ""


@dataclass
class NotableHighlight:
    """A detected anomaly, record, standout milestone, or dramatic turning point."""
    category: str  # "hat_trick", "milestone", "brick_wall", "comeback", "physical", "special_teams"
    title: str
    description: str
    significance: str = "MEDIUM"  # "HIGH", "MEDIUM", "LOW"
    entity: str = ""  # Player name or team abbreviation
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatsValidationReport:
    """Audit report reconciling goals, shots, saves, and period math."""
    is_valid: bool = True
    goals_reconciled: bool = True
    shots_reconciled: bool = True
    period_math_reconciled: bool = True
    audit_notes: List[str] = field(default_factory=list)


@dataclass
class PeriodFlowAnalysis:
    """Deep analysis of lead changes, ties, and period-by-period score trajectory."""
    lead_changes: int = 0
    ties_count: int = 0
    max_winner_lead: int = 0
    max_loser_lead: int = 0
    score_after_p1: str = "0-0"
    score_after_p2: str = "0-0"
    score_entering_p3: str = "0-0"
    winner_trailed_entering_p3: bool = False
    game_script_type: str = "REGULATION"  # WIRE_TO_WIRE, CONTROLLED_LATE_PUSH, TRUE_THRILLER, COMEBACK_WIN, BLOWOUT, WALK_OFF_OT_SO
    script_summary: str = ""


@dataclass
class DefensiveEffortProfile:
    """Team-level defensive effort, shot-blocking sacrifice, and turnover containment."""
    team_abbrev: str
    blocked_shots: int = 0
    blocks_rating: str = "DISCIPLINED"  # SACRIFICIAL (>=18), DISCIPLINED (10-17), POROUS (<8)
    hits: int = 0
    hits_rating: str = "MODERATE"  # PHYSICAL_FORCE (>=25), MODERATE (14-24), CONTAINED (<14)
    giveaways: int = 0
    takeaways: int = 0
    turnover_diff: int = 0
    shots_allowed: int = 0
    shot_suppression: str = "BALANCED"  # SUFFOCATING (<24), BALANCED (24-34), UNDER_SIEGE (>=35)


@dataclass
class GoaliePerformanceProfile:
    """Detailed evaluation of goaltender efficiency, composure, and workload."""
    player_id: int
    name: str
    team_abbrev: str
    decision: Optional[str]
    saves: int = 0
    shots_against: int = 0
    goals_against: int = 0
    save_pct: float = 0.0
    toi: str = "60:00"
    starter: bool = True
    is_shutout: bool = False
    performance_rating: str = "STEADY"  # BRICK_WALL, STEADY, UNDER_SIEGE, VULNERABLE, PULLED
    assessment: str = ""
    xga: float = 0.0
    gsax: float = 0.0
    xsave_pct: float = 0.0
    dsave_pct: float = 0.0
    hd_save_pct: float = 0.0


@dataclass
class TeamSeasonContext:
    """Longitudinal season standings, sample-size stage, and Pythagorean expectation."""
    team_abbrev: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ot_losses: int = 0
    points: int = 0
    points_pct: float = 0.0
    goals_for: int = 0
    goals_against: int = 0
    goal_diff: int = 0
    pythagorean_pct: float = 0.5
    pythagorean_diff: float = 0.0  # points_pct - pythagorean_pct
    season_stage: str = "EARLY_SEASON"  # EARLY_SEASON (<25), MID_SEASON (25-50), PLAYOFF_RACE (>50)
    stage_narrative: str = ""


@dataclass
class LineShiftUnitProfile:
    """On-ice shift performance, sequential momentum, and L5 continuity/change profile for a 3F trio or 2D pair."""
    team_abbrev: str
    unit_type: str  # '3F' or '2D'
    unit_rank: int
    unit_label: str  # 'Line 1', 'Pair 1', etc.
    unit_key: str
    unit_players: str
    toi_sec: int = 0
    toi_formatted: str = "0:00"
    toi_share_pct: float = 0.0
    p1_toi_sec: int = 0
    p2_toi_sec: int = 0
    p3_toi_sec: int = 0
    p3_deployment_shift_pct: float = 0.0
    stints_count: int = 0
    avg_stint_sec: float = 0.0
    cf: int = 0
    ca: int = 0
    cf_pct: float = 50.0
    ff: int = 0
    fa: int = 0
    ff_pct: float = 50.0
    sf: int = 0
    sa: int = 0
    sf_pct: float = 50.0
    gf: int = 0
    ga: int = 0
    goal_diff: int = 0
    ozs_pct: float = 50.0
    dominant_stints: int = 0
    pinned_stints: int = 0
    max_consec_positive_stints: int = 0
    p1_cf_pct: float = 50.0
    p2_cf_pct: float = 50.0
    p3_cf_pct: float = 50.0
    line_impact_score: float = 50.0
    performance_tier: str = "STEADY"
    line_change_status: str = "ESTABLISHED_UNIT"
    l5_exact_unit_games: int = 0
    l5_baseline_cf_pct: float = 50.0
    delta_cf_pct_vs_l5: float = 0.0
    delta_impact_vs_l5: float = 0.0
    l5_trend_verdict: str = "ESTABLISHED_STEADY"
    line_story_summary: str = ""


@dataclass
class TeamShiftAndTrendProfile:
    """Team-level L5 rolling form, schedule adversity, full-season KPIs, and line shift deployment summary."""
    team_abbrev: str
    opponent_abbrev: str = ""
    home_away: str = "HOME"
    rest_days: int = 3
    is_back_to_back: bool = False
    opponent_rest_days: int = 3
    rest_advantage: int = 0
    consec_road_games: int = 0
    consec_home_games: int = 0
    season_gp: int = 0
    season_wp: float = 50.0
    season_cf_pct: float = 50.0
    season_ff_pct: float = 50.0
    season_pythagorean_pe: float = 50.0
    season_rpi: float = 50.0
    l5_games: int = 0
    l5_record: str = "0-0"
    l5_gf_per_game: float = 0.0
    l5_ga_per_game: float = 0.0
    l5_goal_diff: int = 0
    l5_cf_pct: float = 50.0
    l5_ff_pct: float = 50.0
    l5_sf_pct: float = 50.0
    game_cf_pct: float = 50.0
    game_vs_l5_cf_delta: float = 0.0
    l5_form_status: str = "BALANCED"
    has_shift_data: bool = False
    line_continuity_pct: float = 100.0
    new_lines_deployed_count: int = 0
    in_game_line_blender: bool = False
    forward_lines: List[LineShiftUnitProfile] = field(default_factory=list)
    defense_pairs: List[LineShiftUnitProfile] = field(default_factory=list)
    top_line: Optional[LineShiftUnitProfile] = None
    best_line: Optional[LineShiftUnitProfile] = None
    worst_line: Optional[LineShiftUnitProfile] = None
    top_pair: Optional[LineShiftUnitProfile] = None


@dataclass
class EditorialSkillBrief:
    """Structured output from one specialist analytical skill."""
    skill_name: str
    headline_insight: str
    significance_score: float = 50.0
    narrative_points: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EditorialDossier:
    """Synthesized multi-skillset editorial brief guiding the LLM and algorithmic narrators."""
    primary_archetype: str = "BALANCED_CONTEST"
    archetype_label: str = "Competitive Regulation Matchup"
    archetype_rationale: str = ""
    skill_briefs: Dict[str, EditorialSkillBrief] = field(default_factory=dict)
    synthesized_brief_text: str = ""


@dataclass
class AnalyzedGame:
    """Full analytical synthesis of a single game with guardrails and deep contextual tracking."""
    game_id: int
    game_date: str
    game_state: str
    period_type: str  # REG, OT, SO
    venue: str
    away_team: TeamGameStats
    home_team: TeamGameStats
    winner: TeamGameStats
    loser: TeamGameStats
    season: int = 20262027
    is_upset: bool = False
    is_blowout: bool = False
    is_one_goal_game: bool = False
    scoring_plays: List[ScoringPlay] = field(default_factory=list)
    three_stars: List[ThreeStar] = field(default_factory=list)
    away_skaters: List[SkaterStats] = field(default_factory=list)
    home_skaters: List[SkaterStats] = field(default_factory=list)
    away_goalies: List[GoalieStats] = field(default_factory=list)
    home_goalies: List[GoalieStats] = field(default_factory=list)
    notables: List[NotableHighlight] = field(default_factory=list)
    headline: str = ""
    recap_narrative: str = ""
    tactical_summary: str = ""
    full_report_markdown: str = ""
    # Enhanced guardrails & analytical profiles
    validation_report: Optional[StatsValidationReport] = None
    period_flow: Optional[PeriodFlowAnalysis] = None
    away_defense_profile: Optional[DefensiveEffortProfile] = None
    home_defense_profile: Optional[DefensiveEffortProfile] = None
    winning_goalie_profile: Optional[GoaliePerformanceProfile] = None
    losing_goalie_profile: Optional[GoaliePerformanceProfile] = None
    away_season_context: Optional[TeamSeasonContext] = None
    home_season_context: Optional[TeamSeasonContext] = None
    away_shift_trend_profile: Optional[TeamShiftAndTrendProfile] = None
    home_shift_trend_profile: Optional[TeamShiftAndTrendProfile] = None
    editorial_dossier: Optional[EditorialDossier] = None
    xg_report: Optional[Any] = None
    generator_model: str = "algorithmic-guardrails-v1"


@dataclass
class AnalyzedSlate:
    """Collective synthesis across an entire day or slate of games."""
    date: str
    total_games: int
    total_goals: int
    avg_goals_per_game: float
    home_wins: int
    away_wins: int
    ot_so_games: int
    games: List[AnalyzedGame] = field(default_factory=list)
    slate_three_stars: List[ThreeStar] = field(default_factory=list)
    top_performers: List[Dict[str, Any]] = field(default_factory=list)
    slate_notables: List[NotableHighlight] = field(default_factory=list)
    executive_headline: str = ""
    executive_summary: str = ""
    generator_model: str = "algorithmic-guardrails-v1"

