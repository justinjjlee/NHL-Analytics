"""Game Recaps & AI Narratives View for Rinklytics Dashboard.

Features:
- Historical Date Selector: Calendar date picker + pre-compiled Lakehouse archive browser.
- Instant Cached Retrieval: Seamlessly loads historical slates from `jjerlee/nhl-analytics`.
- Verified Game Scripts: Audits period-by-period flow (no false 'thrillers').
- Goaltender & Defensive Sacrifice: Evaluates save composure, blocked shots, and turnover margins.
- Games-Played Sensitivity: Distinguishes early-season noise (<25 GP) from mid-season identities and late-season playoff races.
- Pythagorean Expectations: Identifies teams over/underperforming their underlying goal differentials.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st

try:
    from dev.agent.scoreboard.agent import ScoreboardAgent
    from dev.agent.scoreboard.dataset_store import HFScoreboardStore
    from dev.agent.scoreboard.env_utils import is_local_streamlit, discover_model_name
    from dev.agent.scoreboard.narrator import expand_first_mention_abbreviations, format_milestone_badge
    from dev.agent.scoreboard.skills import ShiftAndTrendDataProvider
except ImportError:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dev.agent.scoreboard.agent import ScoreboardAgent
    from dev.agent.scoreboard.dataset_store import HFScoreboardStore
    from dev.agent.scoreboard.env_utils import is_local_streamlit, discover_model_name
    from dev.agent.scoreboard.narrator import expand_first_mention_abbreviations, format_milestone_badge
    from dev.agent.scoreboard.skills import ShiftAndTrendDataProvider

try:
    from exe.data_loader import load_scoreboard_box_and_odds
except ImportError:
    try:
        from app.exe.data_loader import load_scoreboard_box_and_odds
    except ImportError:
        load_scoreboard_box_and_odds = None


def format_engine_badge(model_name: str) -> str:
    """Format human-readable badge for the active narrative engine writing out the exact model."""
    if not model_name:
        return "Algorithmic Guardrails (v1)"
    m = str(model_name).strip()
    m_low = m.lower()

    if "hf-space" in m_low or "zerogpu" in m_low or "rinklytics" in m_low:
        sub = m.replace("hf-space/", "").replace("zerogpu/", "").strip()
        if sub and sub.lower() not in ("rinklytics", "space", "hf-space", "default"):
            return f"Hugging Face Space - Rinklytics ({sub})"
        active_model = discover_model_name("nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8")
        return f"Hugging Face Space - Rinklytics ({active_model})"
    elif "lm-studio" in m_low:
        clean_name = m.replace("lm-studio/", "").strip()
        return f"LM Studio ({clean_name})"
    elif "remote-endpoint" in m_low:
        clean_name = m.replace("remote-endpoint/", "").strip()
        return f"Remote Endpoint ({clean_name})"
    elif "llama" in m_low:
        return f"Hugging Face Router ({m})"
    elif "algorithmic" in m_low:
        return "Algorithmic Guardrails (v1)"
    else:
        return f"{m}"


@st.cache_resource
def get_store() -> HFScoreboardStore:
    """Singleton cached store connecting to Hugging Face dataset."""
    return HFScoreboardStore()


def _infer_season_label_from_date(d_str: str) -> str:
    """Fallback helper to map YYYY-MM-DD to NHL season string like '2025 - 2026'."""
    try:
        dt = datetime.strptime(d_str[:10], "%Y-%m-%d")
        start_yr = dt.year if dt.month >= 9 else dt.year - 1
        return f"{start_yr} - {start_yr + 1}"
    except Exception:
        return "Unknown Season"


@st.cache_data(ttl=600)
def get_available_game_dates_with_counts() -> tuple[list[str], dict[str, int], dict[str, list[str]], dict[str, str]]:
    """
    Returns:
      - sorted_dates_desc: All valid game dates (YYYY-MM-DD) sorted descending
      - game_counts_by_date: Mapping of YYYY-MM-DD -> completed game count
      - seasons_to_dates: Ordered mapping of Season ('YYYY - YYYY') -> descending list of YYYY-MM-DD
      - date_to_season: Mapping of YYYY-MM-DD -> Season ('YYYY - YYYY')
    """
    game_counts: dict[str, int] = {}
    date_to_season: dict[str, str] = {}

    if load_scoreboard_box_and_odds is not None:
        try:
            box_df, _, available_dates = load_scoreboard_box_and_odds()
            if box_df is not None and not box_df.empty and "date" in box_df.columns:
                date_strs = pd.to_datetime(box_df["date"]).dt.strftime("%Y-%m-%d")
                gid_col = "gameid" if "gameid" in box_df.columns else ("gameId" if "gameId" in box_df.columns else None)
                if gid_col:
                    tmp = box_df.assign(_d=date_strs)
                    counts = tmp.groupby("_d")[gid_col].nunique()
                    first_gid = tmp.groupby("_d")[gid_col].first()
                    for d_str, cnt in counts.items():
                        if int(cnt) > 0:
                            d_key = str(d_str)
                            game_counts[d_key] = int(cnt)
                            gid_str = str(first_gid.get(d_str, ""))
                            if len(gid_str) >= 4 and gid_str[:4].isdigit():
                                sy = int(gid_str[:4])
                                date_to_season[d_key] = f"{sy} - {sy + 1}"
                            else:
                                date_to_season[d_key] = _infer_season_label_from_date(d_key)
                else:
                    counts = (box_df.assign(_d=date_strs).groupby("_d").size() // 2).clip(lower=1)
                    for d_str, cnt in counts.items():
                        if int(cnt) > 0:
                            d_key = str(d_str)
                            game_counts[d_key] = int(cnt)
                            date_to_season[d_key] = _infer_season_label_from_date(d_key)
            elif available_dates is not None and len(available_dates) > 0:
                for d_obj in available_dates:
                    d_str = d_obj.strftime("%Y-%m-%d") if hasattr(d_obj, "strftime") else str(d_obj)[:10]
                    game_counts[d_str] = game_counts.get(d_str, 1)
                    date_to_season[d_str] = _infer_season_label_from_date(d_str)
        except Exception:
            pass

    if not game_counts:
        try:
            box_dir = PROJECT_ROOT / "latest" / "box"
            if box_dir.exists():
                for csv_file in sorted(box_dir.glob("*_box.csv"), reverse=True):
                    df_b = pd.read_csv(csv_file, usecols=lambda c: c in ("gameDate", "date", "gameId", "gameid"))
                    d_col = "gameDate" if "gameDate" in df_b.columns else ("date" if "date" in df_b.columns else None)
                    gid_col = "gameid" if "gameid" in df_b.columns else ("gameId" if "gameId" in df_b.columns else None)
                    if d_col and not df_b.empty:
                        d_strs = pd.to_datetime(df_b[d_col]).dt.strftime("%Y-%m-%d")
                        if gid_col:
                            tmp = df_b.assign(_d=d_strs)
                            counts = tmp.groupby("_d")[gid_col].nunique()
                            first_gid = tmp.groupby("_d")[gid_col].first()
                            for d_str, cnt in counts.items():
                                if int(cnt) > 0:
                                    d_key = str(d_str)
                                    game_counts[d_key] = int(cnt)
                                    gid_str = str(first_gid.get(d_str, ""))
                                    if len(gid_str) >= 4 and gid_str[:4].isdigit():
                                        sy = int(gid_str[:4])
                                        date_to_season[d_key] = f"{sy} - {sy + 1}"
                                    else:
                                        date_to_season[d_key] = _infer_season_label_from_date(d_key)
                        else:
                            counts = (df_b.assign(_d=d_strs).groupby("_d").size() // 2).clip(lower=1)
                            for d_str, cnt in counts.items():
                                if int(cnt) > 0:
                                    d_key = str(d_str)
                                    game_counts[d_key] = int(cnt)
                                    date_to_season[d_key] = _infer_season_label_from_date(d_key)
        except Exception:
            pass

    if not game_counts:
        try:
            store = get_store()
            for d_str in store.get_available_historical_dates():
                d_key = str(d_str)
                game_counts[d_key] = 1
                date_to_season[d_key] = _infer_season_label_from_date(d_key)
        except Exception:
            pass

    if not game_counts:
        try:
            agent = ScoreboardAgent()
            latest = agent.get_latest_date()
            game_counts[latest] = 1
            date_to_season[latest] = _infer_season_label_from_date(latest)
        except Exception:
            fallback = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            game_counts[fallback] = 1
            date_to_season[fallback] = _infer_season_label_from_date(fallback)

    sorted_dates = sorted(game_counts.keys(), reverse=True)
    seasons_map: dict[str, list[str]] = {}
    for d_str in sorted_dates:
        s_label = date_to_season.get(d_str) or _infer_season_label_from_date(d_str)
        date_to_season[d_str] = s_label
        seasons_map.setdefault(s_label, []).append(d_str)

    sorted_seasons = sorted(seasons_map.keys(), reverse=True)
    seasons_to_dates = {s: seasons_map[s] for s in sorted_seasons}
    return sorted_dates, game_counts, seasons_to_dates, date_to_season


@st.cache_data(ttl=600)
def get_latest_table_game_date() -> str:
    """Finds the most recent completed game date available in the primary box scores table."""
    sorted_dates, _, _, _ = get_available_game_dates_with_counts()
    return sorted_dates[0]


def run_generation_pipeline(target_date: str, store: HFScoreboardStore, is_local: bool) -> bool:
    """Executes full slate analysis, LLM journalistic synthesis, and stores recaps."""
    with st.spinner(f"Analyzing NHL slate and synthesizing narratives for {target_date}..."):
        agent = ScoreboardAgent()
        if is_local:
            is_healthy, lm_model, alarm_err = agent.narrator.check_lm_studio_health()
            if not is_healthy and agent.narrator.model_mode in ("lm-studio", "auto"):
                st.warning(
                    f"⚠️ **LM Studio Offline**: Cannot reach local server at `{agent.narrator.lm_studio_url}` ({alarm_err}). "
                    "Recaps will be generated using the deterministic Algorithmic Guardrail Engine."
                )
        else:
            import os
            is_space_up, space_name, space_note = agent.narrator.check_hf_space_health()
            if agent.narrator._can_use_zerogpu():
                active_space_model = discover_model_name(getattr(agent.narrator, "zerogpu_model_id", "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"))
                st.info(f"⚡ **Hugging Face Space Runtime ({space_name})**: Connected to ZeroGPU. Synthesizing verified recaps via in-process `{active_space_model}`.")
            else:
                custom_endpoint = (
                    agent.narrator.hf_llm_endpoint
                    or os.environ.get("HF_LLM_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("LLM_BASE_URL")
                )
                if custom_endpoint:
                    st.info(f"🌐 **Remote LLM Endpoint Connected**: Synthesizing recaps via `{custom_endpoint}`.")
                else:
                    st.info(
                        f"ℹ️ **Production Runtime ({space_name})**: Connected. "
                        "Reading pre-generated LLM recaps from dataset; on-demand runs use the deterministic Algorithmic Guardrail Engine (set `HF_LLM_URL` in Space Secrets for remote LLM inference)."
                    )

        try:
            analyzed_slate = agent.analyze_slate(target_date)
            analyzed_games = analyzed_slate.games
        except Exception as e:
            st.error(f"Error fetching data for {target_date}: {e}")
            return False

        if not analyzed_games:
            st.warning(
                f"ℹ️ No completed NHL games were found for **{target_date}**. "
                "This date may have been an off-day, holiday break, or off-season."
            )
            return False

        # Record to Hugging Face dataset store
        slate_model = getattr(analyzed_slate, "generator_model", "algorithmic-guardrails-v1")
        game_model = (
            getattr(analyzed_games[0], "generator_model", "algorithmic-guardrails-v1")
            if analyzed_games
            else "algorithmic-guardrails-v1"
        )
        store.record_slate(analyzed_slate, generator_model=slate_model)
        store.record_games(analyzed_games, generator_model=game_model)

        # Push to Hugging Face if HF_TOKEN is present
        if store.token:
            sync_res = store.sync_to_huggingface()
            if sync_res.get("success"):
                st.toast("☁️ Synced to Hugging Face dataset `insights/scoreboard/` (jjerlee/nhl-analytics)!")

        st.success(f"Successfully generated recaps for {target_date}!")
        return True


def render_game_recaps_view():
    st.markdown(
        '<h1 style="display:flex; align-items:center; gap:12px; margin-bottom:0.2rem;">'
        'Game Recaps '
        '<span style="background-color:#FF4B4B; color:white; font-size:0.42em; font-weight:700; '
        'padding:3px 10px; border-radius:999px; letter-spacing:0.06em; text-transform:uppercase; '
        'vertical-align:middle;">Beta</span>'
        '</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Daily game recaps, line-shift chemistry, and analytical summaries (Trial Mode).")
    st.info(
        "ℹ️ **Beta Trial & AI Editorial Disclosure:** These game recaps and analytical insights are generated by a "
        "Large Language Model (LLM) using actual tabulated NHL box-score, play-by-play, and shift telemetry. "
        "While grounded in verified statistical records, the editorial tone, contextual emphasis, and narrative "
        "takeaways are experimental and may differ from unanimous or majority observations by live viewers and beat reporters."
    )

    store = get_store()
    (
        available_game_dates,
        game_counts_by_date,
        seasons_to_dates,
        date_to_season,
    ) = get_available_game_dates_with_counts()
    latest_table_date = available_game_dates[0]
    recorded_dates_set = set(store.get_available_historical_dates())

    # 1. Initialize and strictly snap selected date to an available game date
    if (
        "recap_target_date" not in st.session_state
        or st.session_state.recap_target_date not in available_game_dates
    ):
        st.session_state.recap_target_date = latest_table_date

    target_date = st.session_state.recap_target_date
    season_options = list(seasons_to_dates.keys())
    current_season = date_to_season.get(target_date, season_options[0])
    if current_season not in season_options:
        current_season = season_options[0]
    current_season_idx = season_options.index(current_season)

    def _format_game_date_option(d_str: str) -> str:
        try:
            return datetime.strptime(d_str, "%Y-%m-%d").strftime("%a, %b %d, %Y")
        except Exception:
            return d_str

    is_local = is_local_streamlit()
    is_recorded = store.is_slate_recorded(target_date)
    slate_record = store.get_slate_record(target_date) if is_recorded else None
    game_records = store.get_games_for_date(target_date) if is_recorded else []
    has_narratives = bool(
        is_recorded and slate_record and game_records and any(g.get("narrative_recap") for g in game_records)
    )

    trigger_generate = False

    # -------------------------------------------------------------------------
    # 2. Season Selector + Date Selector (restricted strictly to valid game dates)
    # -------------------------------------------------------------------------
    if is_local:
        col_season, col_date, col_btn = st.columns([1.1, 2.2, 1.0], vertical_alignment="bottom")
        with col_season:
            selected_season = st.selectbox(
                "Select Season",
                options=season_options,
                index=current_season_idx,
                help="Filter available game dates by NHL season.",
                key="recap_season_selectbox",
            )
            if selected_season != current_season:
                st.session_state.recap_target_date = seasons_to_dates[selected_season][0]
                st.rerun()

        season_dates = seasons_to_dates.get(selected_season, available_game_dates)
        season_date_idx = season_dates.index(target_date) if target_date in season_dates else 0

        with col_date:
            selected_date_str = st.selectbox(
                "Select Game Date",
                options=season_dates,
                index=season_date_idx,
                format_func=_format_game_date_option,
                help="Only dates with completed NHL games in the selected season are shown.",
                key="recap_valid_game_date_selectbox",
            )
            if selected_date_str != st.session_state.recap_target_date:
                st.session_state.recap_target_date = selected_date_str
                st.rerun()

        with col_btn:
            btn_label = "Force Re-Analyze" if has_narratives else "Generate Recaps"
            btn_type = "secondary" if has_narratives else "primary"
            btn_help = (
                "Re-analyze current date with LLM and sync to Hugging Face dataset (Local Dev Only)"
                if has_narratives
                else "Synthesize and save LLM game recaps for this date (Local Dev Only)"
            )
            trigger_generate = st.button(btn_label, type=btn_type, use_container_width=True, help=btn_help)
    else:
        col_season, col_date = st.columns([1.2, 2.8], vertical_alignment="bottom")
        with col_season:
            selected_season = st.selectbox(
                "Select Season",
                options=season_options,
                index=current_season_idx,
                help="Filter available game dates by NHL season.",
                key="recap_season_selectbox",
            )
            if selected_season != current_season:
                st.session_state.recap_target_date = seasons_to_dates[selected_season][0]
                st.rerun()

        season_dates = seasons_to_dates.get(selected_season, available_game_dates)
        season_date_idx = season_dates.index(target_date) if target_date in season_dates else 0

        with col_date:
            selected_date_str = st.selectbox(
                "Select Game Date",
                options=season_dates,
                index=season_date_idx,
                format_func=_format_game_date_option,
                help="Only dates with completed NHL games in the selected season are shown.",
                key="recap_valid_game_date_selectbox",
            )
            if selected_date_str != st.session_state.recap_target_date:
                st.session_state.recap_target_date = selected_date_str
                st.rerun()

    # Trigger generation pipeline if local button was clicked
    if trigger_generate:
        success = run_generation_pipeline(target_date, store, is_local)
        if success:
            store._load_tables()
            st.rerun()

    # -------------------------------------------------------------------------
    # 3. Check Narrative Generation Status (Warning Sign if Not Generated)
    # -------------------------------------------------------------------------
    if not has_narratives:
        st.warning(f"⚠️ Game recaps and analytical narratives have not been generated for **{_format_game_date_option(target_date)}**.")
        if is_local:
            st.info("💡 You are running locally (`local_streamlit=True`). Click **Generate Recaps** next to the date selector above to run local editorial synthesis.")
        else:
            st.info("ℹ️ Production game recaps are published automatically on a daily schedule once the evening slate is finalized.")
        return

    st.success(f"Loaded recaps for **{target_date}**.")

    # -------------------------------------------------------------------------
    # 3. Slate Summary KPI Cards
    # -------------------------------------------------------------------------
    slate_gen = slate_record.get("generator_model", "algorithmic-guardrails-v1")
    slate_engine_badge = format_engine_badge(slate_gen)
    st.subheader(slate_record.get('slate_headline', f'NHL Slate Overview — {target_date}'))
    st.caption(f"**Data Quality:** Reconciled & Audited | **Editorial Engine:** {slate_engine_badge}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Games", slate_record.get("total_games", 0))
    m2.metric(
        "Home vs Road Wins",
        f"{slate_record.get('home_wins', 0)} - {slate_record.get('away_wins', 0)}",
    )
    m3.metric("Overtime / Shootouts", slate_record.get("overtime_games", 0))

    # --- THREE STARS OF THE NIGHT PODIUM ---
    three_stars_json = slate_record.get("three_stars_json")
    if three_stars_json:
        try:
            stars = json.loads(three_stars_json)
            if stars:
                st.markdown("### Three Stars of the Night")
                star_cols = st.columns(len(stars))
                for i, star in enumerate(stars):
                    with star_cols[i]:
                        medal = "🥇" if star.get("star_num") == 1 else ("🥈" if star.get("star_num") == 2 else "🥉")
                        st.markdown(f"**{medal} #{star.get('star_num')} Star: {star.get('name')}**")
                        st.caption(f"{star.get('team')} • {star.get('position')} | {star.get('stat_summary')}")
        except Exception:
            pass

    # --- SLATE FULL NARRATIVE EXPANDER ---
    slate_narrative = slate_record.get("slate_narrative")
    if slate_narrative:
        with st.expander("League Slate Media Review", expanded=False):
            st.markdown(expand_first_mention_abbreviations(slate_narrative))

    st.divider()

    # -------------------------------------------------------------------------
    # 4. Individual Game Recaps with Verified Guardrails
    # -------------------------------------------------------------------------
    st.subheader(f"Game Recaps ({len(game_records)} Games on {target_date})")

    script_badges = {
        "WIRE_TO_WIRE": "Wire-to-Wire Command",
        "CONTROLLED_LATE_PUSH": "Controlled Late Push",
        "COMEBACK_WIN": "3rd-Period Comeback",
        "TRUE_THRILLER": "See-Saw Thriller",
        "WALK_OFF_OT_SO": "Sudden-Death Walk-Off",
        "BLOWOUT": "Offensive Rout",
        "REGULATION": "Regulation",
    }

    dec_map = {"REG": "Final", "OT": "Final/OT", "SO": "Final/SO"}

    def _clean_rating(val: Any) -> str:
        if not val:
            return ""
        return str(val).replace("_", " ").title()

    for game in game_records:
        script_key = game.get("game_script_type", "REGULATION")
        badge = script_badges.get(script_key, "Regulation")
        period_ending = game.get("period_ending", "REG")
        dec_label = dec_map.get(period_ending, period_ending)

        header_label = (
            f"{badge} • {game.get('away_team')} {game.get('away_score')} @ "
            f"{game.get('home_team')} {game.get('home_score')} "
            f"({dec_label}) — {game.get('headline')}"
        )

        with st.expander(header_label):
            st.markdown(f"### {game.get('headline')}")
            game_gen = game.get("generator_model", "algorithmic-guardrails-v1")
            game_engine_badge = format_engine_badge(game_gen)
            st.caption(f"**Game Script:** {badge} | **Engine:** {game_engine_badge}")

            # Game Narrative Story (with first-mention abbreviations expanded)
            st.markdown(expand_first_mention_abbreviations(game.get("narrative_recap", "")))

            # Standout Milestones
            milestones_str = game.get("milestones_json")
            if milestones_str:
                try:
                    milestones = json.loads(milestones_str)
                    if milestones:
                        st.markdown("#### Detected Milestones")
                        for m in milestones:
                            st.markdown(f"- {format_milestone_badge(m.get('category', ''), m.get('title', ''), m.get('description', ''))}")
                except Exception:
                    pass

            # Optional Statistical Breakdown & Box Score Tables
            show_tables = st.checkbox(
                "Show Detailed Box Score & Analytical Tables",
                value=False,
                key=f"show_tables_{game.get('game_id')}",
            )
            if show_tables:
                # A. Goaltending Breakdown Table
                goalie_json = game.get("goalie_performance_json")
                if goalie_json:
                    try:
                        g_data = json.loads(goalie_json)
                        if g_data:
                            st.markdown("#### Goaltending Performance")
                            g_rows = []
                            for role in ("winner", "loser"):
                                if role in g_data:
                                    p = g_data[role]
                                    g_rows.append({
                                        "Goaltender": p.get("name"),
                                        "Team": p.get("team"),
                                        "Dec": p.get("decision"),
                                        "Saves / Shots": f"{p.get('saves')}/{p.get('shots_against')}",
                                        "SV%": f"{p.get('save_pct', 0.0)*100:.1f}%",
                                        "Workload Rating": _clean_rating(p.get("rating")),
                                        "Assessment": p.get("assessment"),
                                    })
                            if g_rows:
                                st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass

                # B. Defensive Sacrifice & Physicality Table
                def_json = game.get("defensive_effort_json")
                if def_json:
                    try:
                        d_data = json.loads(def_json)
                        if "away" in d_data and "home" in d_data:
                            st.markdown("#### Defensive Effort & Physicality")
                            a = d_data["away"]
                            h = d_data["home"]
                            def_rows = [
                                {
                                    "Metric": "Blocked Shots",
                                    f"{game.get('away_team')}": f"{a.get('blocked_shots')} ({_clean_rating(a.get('blocks_rating'))})",
                                    f"{game.get('home_team')}": f"{h.get('blocked_shots')} ({_clean_rating(h.get('blocks_rating'))})",
                                },
                                {
                                    "Metric": "Body Checks (Hits)",
                                    f"{game.get('away_team')}": f"{a.get('hits')} ({_clean_rating(a.get('hits_rating'))})",
                                    f"{game.get('home_team')}": f"{h.get('hits')} ({_clean_rating(h.get('hits_rating'))})",
                                },
                                {
                                    "Metric": "Giveaways (Turnovers)",
                                    f"{game.get('away_team')}": str(a.get("giveaways")),
                                    f"{game.get('home_team')}": str(h.get("giveaways")),
                                },
                                {
                                    "Metric": "Takeaways",
                                    f"{game.get('away_team')}": str(a.get("takeaways")),
                                    f"{game.get('home_team')}": str(h.get("takeaways")),
                                },
                                {
                                    "Metric": "Shots Allowed",
                                    f"{game.get('away_team')}": f"{a.get('shots_allowed')} ({_clean_rating(a.get('shot_suppression'))})",
                                    f"{game.get('home_team')}": f"{h.get('shots_allowed')} ({_clean_rating(h.get('shot_suppression'))})",
                                },
                            ]
                            st.dataframe(pd.DataFrame(def_rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass

                # C. Season Standings & Pythagorean Analytics Table
                season_json = game.get("season_standings_json")
                if season_json:
                    try:
                        s_data = json.loads(season_json)
                        if s_data:
                            st.markdown("#### Season Progression & Pythagorean Expectation")
                            s_rows = []
                            for side in ("away", "home"):
                                if side in s_data:
                                    ctx = s_data[side]
                                    s_rows.append({
                                        "Team": ctx.get("team"),
                                        "GP": ctx.get("gp"),
                                        "Record": ctx.get("record"),
                                        "PTS": ctx.get("pts"),
                                        "PTS%": f"{ctx.get('pts_pct', 0.0)*100:.1f}%",
                                        "GF": ctx.get("gf"),
                                        "GA": ctx.get("ga"),
                                        "Diff": f"{ctx.get('goal_diff', 0):+d}",
                                        "Pyth Exp %": f"{ctx.get('pyth_pct', 0.0)*100:.1f}%",
                                        "Luck/Variance": f"{ctx.get('pyth_diff', 0.0)*100:+.1f}%",
                                        "Season Stage": _clean_rating(ctx.get("stage")),
                                    })
                            if s_rows:
                                st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)
                                stage_desc = s_data.get("away", {}).get("stage_narrative", "")
                                if stage_desc:
                                    st.caption(f"**Sample-Size Context:** {stage_desc}")
                    except Exception:
                        pass

                # D. Line Shift Chemistry & Past 5 Games (L5) Trend Table
                try:
                    gid_int = int(game.get("game_id", 0))
                    away_tri = str(game.get("away_team", ""))
                    home_tri = str(game.get("home_team", ""))
                    away_sp = ShiftAndTrendDataProvider.get_team_shift_trend_profile(gid_int, away_tri, home_tri, is_home=False)
                    home_sp = ShiftAndTrendDataProvider.get_team_shift_trend_profile(gid_int, home_tri, away_tri, is_home=True)

                    shift_rows = []
                    for sp in (away_sp, home_sp):
                        if sp and sp.has_shift_data:
                            for u in sp.forward_lines[:4] + sp.defense_pairs[:3]:
                                shift_rows.append({
                                    "Team": sp.team_abbrev,
                                    "Unit": u.unit_label,
                                    "Skaters": u.unit_players,
                                    "TOI": f"{u.toi_formatted} ({u.toi_share_pct:.1f}%)",
                                    "P3 Shift": f"{u.p3_deployment_shift_pct:+.1f}%",
                                    "CF-CA (CF%)": f"{u.cf}-{u.ca} ({u.cf_pct:.1f}%)",
                                    "SOG": f"{u.sf}-{u.sa}",
                                    "GF-GA": f"{u.gf}-{u.ga}",
                                    "Line Status (vs L5)": _clean_rating(u.line_change_status),
                                    "Δ CF% vs L5": f"{u.delta_cf_pct_vs_l5:+.1f}% (L5: {u.l5_baseline_cf_pct:.1f}%)",
                                    "L5 Trend Verdict": _clean_rating(u.l5_trend_verdict),
                                    "Impact": f"{u.line_impact_score:.1f} ({_clean_rating(u.performance_tier)})",
                                })
                    if shift_rows:
                        st.markdown("#### Line Shift Chemistry & Past 5 Games (L5) Trends")
                        st.dataframe(pd.DataFrame(shift_rows), use_container_width=True, hide_index=True)
                        st.caption(
                            "**Table Legend:** **TOI** = Time on Ice (% share of team unit time) • "
                            "**P3 Shift** = Period 3 deployment share change vs. Periods 1–2 • "
                            "**CF-CA (CF%)** = Corsi For vs. Corsi Against (total shot-attempt share) • "
                            "**SOG** = Shots on Goal For–Against • **GF-GA** = On-Ice Goals For–Against • "
                            "**L5** = Rolling Last 5 Games baseline."
                        )
                        for sp in (away_sp, home_sp):
                            if sp and sp.has_shift_data and sp.best_line:
                                st.caption(
                                    f"**{sp.team_abbrev} Standout Unit:** "
                                    f"{expand_first_mention_abbreviations(sp.best_line.line_story_summary)}"
                                )
                except Exception:
                    pass


if __name__ == "__main__" or "page" in locals():
    render_game_recaps_view()
