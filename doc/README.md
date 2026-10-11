# NHL Analytics Documentation Knowledge Base

Welcome to the **NHL Analytics (Rinklytics)** knowledge base. This directory documents the system capabilities, data architectures, statistical findings, and analytical methodologies developed across the platform.

---

## Documentation Sections

### 1. [API Capabilities & Data Availability](api/README.md)
* **[Overview](api/README.md):** Architecture of the modern NHL API ecosystem (`api-web.nhle.com`, `api.nhle.com/stats/rest/en`, and `assets.nhle.com`), deprecation of legacy endpoints, authentication, and rate limiting.
* **[Data Views & Availability Matrix](api/data_views_and_capabilities.md):** Structural view of all data views, granularities (event, shift, player tracking), and temporal availability (real-time vs. historical horizons vs. NHL Edge 2021+).
* **[Endpoints Reference](api/endpoints_reference.md):** Detailed technical specifications for every available endpoint across games, play-by-play, shift charts, player careers, entry draft, betting odds, and SVG media assets.
* **[Coordinate Systems & Data Dictionary](api/data_dictionary_and_coords.md):** Official NHL ice rink coordinate space ($x \in [-100, 100]$, $y \in [-42.5, 42.5]$), home/away rink orientation normalization, event codes, and 10-digit Game ID decoding.
* **[Ingestion Pipelines](api/ingestion_pipelines.md):** How the platform's automated Python pipelines (`src/data/apinhle/`) ingest, process, and persist data to `latest/` and Databricks Unity Catalog volumes.

### 2. [Box Score ETL & Opponent Pairing](box/README.md)
* **[Box Score Architecture](box/README.md):** How raw game schedules and boxscores are queried, unpacked, and symmetrically matched using the bilateral opponent pairing engine (`df_unpack`).
* **Temporal Rest & Fatigue Deltas:** Measuring physical fatigue ($\Delta t_{\text{rest}}$), back-to-backs, and forward game indices (`gameIdx_next`).
* **Artifact Catalog:** Schemas for raw schedule files, game results, team box scores, dressed player box stats, and enriched game records.

### 3. [Analytical Measurements & Mathematical Modeling](measurement/README.md)
* **[Measurement Models](measurement/README.md):** Detailed mathematical formulations, algorithmic logic, and Mermaid execution diagrams for platform metrics.
* **Process vs. Outcome:** Shot generation proxies (Corsi and Fenwick differentials and percentages).
* **Quality & Win Expectation:** Pythagorean Expectation ($PE$), 10-game rolling window ($PE_{10}$), and empirical 82-game linear win projections ($\beta_{\text{pyth}}, \alpha_{\text{pyth}}$).
* **Schedule Difficulty Adjustments:** Rating Percentage Index ($RPI$) and NCAA-style Head-to-Head Pairwise Common Opponent Dominance.
* **Streak Disentanglement:** Isolating true quality-adjusted "Hot Streaks" ($\text{RPI}_{\text{opp}} \ge Q_{0.75}$) from schedule-favored "Expected Streaks".
* **Individual Composite Impact:** Player Performance Index ($\text{PPI}$) composite z-score formulation for skaters and goaltenders.

---

## Planned Sections for Future Findings & Capabilities
As ongoing research in `dev/` and visualization modules in `app/` expand, this directory will host:
* **`doc/decision_science/`**: Causal inference and game-theoretic findings (e.g., *Impact of Physical Hits on Momentum and Goal Generation*, *Defensive Shot-Blocking Efficacy*).
* **`doc/player_lifecycle/`**: Amateur-to-professional player development, draft survival analysis, and exceptional cohort trajectory modeling.
* **`doc/market_analytics/`**: Real-time sports betting market efficiency, implied vs. empirical win probabilities, and line movement tracking.
