# Measuring, evaluating, and benchmarking team's success
In this exercise, I examine a number of headline team performance measures of National Hockey League (NHL). The goal of the exercise is to examine usefulness of the traditional measure of team performance translates to success of the team (win totals).

I ask the following headline questions: 

1. What are the components to look for in predicting team's future performace?
2. How headline performance measures are calculated and how they differ
3. Are these baseline predictor have equal or better predictive power than betting lines/odds, thus represeting sufficient information as in efficient market (hypothesis)?
4. Say that these headline indicators (in real-time) are the only measure to predict the outcome, which measures tend to best estimate the outcome of the next game?

This analysis series is based on the data processing flow that I created, from [my NHL Analytics Github repository](https://github.com/justinjoliver/NHL-Analytics).

## Setting up the data
Details on how the play/game level data are pulled and processed are saved in [my Github repository](https://github.com/justinjoliver/NHL-Analytics).

### Data features to consider when creating metrics

Note that 0: loss; 0.5: overtime; 1: outright win

**Goals for and Goals against**
* Level (aggregation)
* Differential: Goals for - Goals Against
* Ratio: Goals for / Goals Against

**Possession change** 
* Level sum (aggregation)

**Penalty**
* Difference: pim own - pim opponent
* Difference: power play count - penalty count (this is less needed with pim information available)

**Range**
* Season summary (to date for real-time evaluations)
* Rolling summary (to evaluate vintage performance at-the-time)
* Recency rolling summary (to evaluate past specific windows)

Priority measure: headline measure best serves predictor of performance.
* Pythagorean Expectation
* [Corsi measurement](https://medium.com/hockey-stats/advanced-hockey-stats-101-corsi-part-1-of-4-29d0a9fb1f95)
* pairwise predictor, use in [college hockey](https://www.uscho.com/faq/pairwise-rankings-explanation/) to combine both stregth of records and strength of schedule.

## Corsi & Fenwick

The [Corsi](https://en.wikipedia.org/wiki/Corsi_(statistic)) measure compares shots made by own team and shots made by its opponents

* Corsi For (CF) = Shot attempts for at even strength: Shots + Blocks + Misses
* Corsi Against (CA) = Shot attempts against at even strength: Shots + Blocks + Misses
* Corsi (C) = CF - CA
* Corsi For % (CF%) = CF / (CF + CA)

The [Fenwick](https://en.wikipedia.org/wiki/Fenwick_(statistic)) measure is similar, in that

$Fenwick = (Shots^{for} + Shots^{for}_{missed}) – (Shots^{against} + Shots^{against}_{missed})$

## Pythagorean Expectation (PE)

Pythagorean Expectation is an aggregate headline measure commonly used in team sports, establishing expectations of teams success based on how much they can score and how much they can defend opponents from scoring. 

Historically, the masure has been extremely successful in benchmarking teams sucess during the regular seasons and during the Stanely Cup Playoff. My [dashboard](https://public.tableau.com/app/profile/justin.l.1253/viz/NHL-PythagoreanExpectation/Dashboard2) shows the historical performance of each team and the season PE going back to 1917.

Up-to-date performance measurement of team's performance,

$\frac{(Goals \; for)^2}{(Goals \; for)^2 + (Goals \; against)^2}$

Measure of entire season as a reference

## Rating Percentage Index (RPI)

More direct measure to compare is to have weighted grade by incorporating opponents record and opponents' opponents record (not ncessarily head-to-head).

1. [Rating Percentage Index](https://en.wikipedia.org/wiki/Rating_percentage_index#:~:text=The%20rating%20percentage%20index%2C%20commonly,and%20its%20strength%20of%20schedule.)
You can be creative, such as weighting home/away game wins; but we use a simple formula

$RPI = (WP * 0.25) + (OWP * 0.50) + (OOWP * 0.25)$

Because we do not have 'season' level, we need rolling (cumulative measure) of expectation

## Rolling Pythagorean Expectation(rPE)

Imagine teams climbing up (hopefully) the trend line as season progress

As season progress (dark to light) teams find their season expected performance (equilibrium). Question is - when and where each team will reach at the end? are seasons decided before All-star period? One way to check is to see how they change over time, specifically in last $n$ games.

The median estimates of the correlation above suggests that teams are more likely to win if their recent performance has been better than they did in previous period (positive changes in rPE). 

This may sound obvious, but this theoretically hints a weak evidence of 'hot streak' - where teams that are starting to perform better tends to continue to perform better (leading to multiple wins in a row).

One factor to account for, an argument against the importance of 'hot streak,' is the performance of extrema's. If certain number of teams are performing extremely poorly and others extremely well, then the $rPE$ and $ΔrPE$ measure would naturally sckewed towards loss/win, respectively. A good example would be teams that suffer since the beginning of the season and continues to struggle throughout the season. This is well documented in the expectation of outcome as well, in [my evaluation of betting market](https://medium.com/@thinkingjustin/better-than-puck-flips-money-line-and-market-expectation-in-nhl-91786fcf6f01).

$\Delta rPE$ shows a stronger relationship with the win at the time (recency) than cumulative win (an overall success), and vice versa with $rPE$. Also, the correlation with $win$ between $\Delta rPE$ and $rPE$ are almost identical.

Above observations hint a couple takeaways
* Recency of performance does not necessarily provides a better predictor than the cumulative representation of team's success up to the time → Argument against the hot-streak.
* Recency of performance provides a better predictor in real-time than longer out horizon → This also changes as the season matures.

rPE can be used as a proxy measure of 'regime' or 'hot streak' - if there is such a thing in sports (hint, hint - this is a hot topic. See [one of the recent studies](https://www.nber.org/papers/w29468)). In technicality, without going off topic, the regime can be confused with random probability in local regime (finite monte hall problem), not a large-sample long duration game. 

The changes in variability over time is something we can check - if those hot streaks tend to 

## Head-to-head: Measurements of Opponents' Opponents

Measure starts with comparing 
1. Head to Head performance
2. Match records against common opponents
3. Majority rule of performance against the common opponents

### (a) Pairwise - head to head win count
Head to head win count rank

## Further thoughts on relative performance - brainstorm

Unlike the college version, in head to head, how do we decide who is clearly better? In head to head comparisons, especially when not many games are played (small sample), it is hard to gauge who is better team head to head.
* Win percentage? (if played more than one game)
* Goals per game? goals against?
* shot / shot blcoked missed

e.g. team 'a' plays team 'x' two times and win once, whereas team 'b' plays the team 'x' once but walk away without a win. is team 'a' clearly better than team 'a'? In these cases, pythagorean expectation can be helpful, or leveraging shot block.

In other cases as well - head to head of clearly better can be most useful with pythagorean expectation.

Idea: use of pythagorean expectation, we can just aggregate all up and evaluate as one point
* In this case, we need a tie-breaker.