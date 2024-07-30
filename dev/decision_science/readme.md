# Decision Science Framework
Work related to analyzing sequence of data. I use `DuckDB` as main source of pulling and simple-processing the sequence panel data for out-of-memory processing.

## Data processing
- [x] Tracking previous activities and events
- [x] Tracking power play and penality kill periods
    - [x] Marking sequence that falls within the penalty time assessed
    - [x] Tracking the penalty periods that cross over periods: by creating continued time stamp over the entire game
    - [x] Tracking penalties that overlaps across teams,
    - [ ] Tracking penalties that are not perfectly overlapping, such as one penalty was followed by another during the first penality being served
    - [ ] Tracking mark from called delayed penalty to stoppage of play (in case of players being pulled)

## Identifying specific trends

### Power Play & Penalty Kill
The power play and penalty kill actions and decisions are made. Following logics can be used to identify those trends based on the logics created within the data.

Note that the actions of `penalty kill` behaviors are reverse of the team marked as `power play` actions.

| Action Type | Actions/Decisions | Captured in the Data |
| --- | --- | --- |
| Power Play | Duration | `penalty_period IS NOT NULL` |
| Power Play | Goal | `eventOnwerTeam_goalscore = temp_pp_lag_team` |