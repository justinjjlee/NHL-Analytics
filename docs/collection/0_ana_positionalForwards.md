# Positional successes in the offensive zone

## Highlights

*	Stanley Cup-winning teams have very specific strategies to generate efficient puck possessions in offensive zone
* Zach Hyman, Edmonton Oilers's new addition, may complement where Connor McDavid had efficiency challenges
* Maple Leaf’s struggle can be explained by its positional dependencies on its top forwards

## Measurement

Observing play-level statistics from 2016-17 to 2020-21 season, I plot over (*colored orange*) and under (*colored blue*) shots-on-goal percentages of each players relative to the league's average based on their position in offensive zone. See details in the methodology section. Along with the first-layer information, location of the event allows me to identify strategic positional patterns for each team and player.

## Anecdotal successful player positions and strategies
I observe three recent Stanley Cup-winning teams: Washington Capitals (2018), St. Louis Blues (2019), and Tampa Bay Lightning (2020 and 2021). Especially, the top forward lines for the teams have distinct approaches that led the teams' successes. The table below notes each team's first lines for forwards during the seasons.

### Top Forward lines for each team

| Teams | Line Combinations |
| --- | :---: | 
| Washington Capitals | Ovechkin - Kuznetsov/Backstrom - Wilson/Oshi |
| St. Louis Blues | Schenn-Schwartz-Tarasenko |
| Tampa Bay Lightening | Point-Palat-Kucherov |

### Capital's stay-in-your-lane approach
Capitals benefit from a reliable goal-scoring from distance by Alex Ovechkin. In order to synergize, other forwards have been narrowly positioned near the goal crease. This effectively generates blocks and follow-through scoring opportunities. 

*Washington Capitals top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_caps_18.gif?raw=true)

### Kitchen-sink approach by St. Louis Blues
Blues in its 2018-2019 contest benefited from the top line's presence and scoring approach. The ability to score was complemented with a heavy presence up front.

*St. Louis Blues top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_stl_2019.gif?raw=true)

### Kucherov we trust
Lightening’s hole-in-one approach leverages up-front presence by forwards and one sharp-shooter. Centered around the top scorer, Nikita Kucherov, Lightening manages to position its forwards to open up opportunities and follow through shots missed or blocked.

*Tampa Bay Lightening top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_tbl_2021.gif?raw=true)

## Oiler's Acquisition of Hyman may complement McDavid
During 2021 offseason, Edmonton Oilers acquired Zach Hyman from Toronto Maple Leafs. Based on the the positional performance as Leafs, I hypothesize that Zach Hyman will complement existing top scorers of the team (Connor McDavid and Leon Draisaitl). Figure below (*right*) shows Connor McDavid’s relative goal scoring success on the rink. McDavid struggles to score from the center while his assists in goals shines (figure *left*). This implies that McDavid’s production can be amplified if he is complemented in the region. An ideal candidate is a physical player who knows how to grind out the game and is not afraid of hard-skating or able to open up the offensive zone.

Assists - McDavid          |  Goals - McDavid
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Connor%20McDavid_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Connor%20McDavid_goal_success.png?raw=true)

While the gap is not apparent for Draisaitl,

Assists - Draisaitl          |  Goals - Draisaitl
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Leon%20Draisaitl_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Leon%20Draisaitl_goal_success.png?raw=true)

As a Leafs’ left winger, Hyman’s point production was not as stellar as Panarin or Gaudreau. However, his presence up front and better-than-average production right under goalie’s nose would complement where McDavid struggles. As Hyman is known to be one of the fastest and hardest skater on ice, it would be exciting to watch the pair skate together.

*Hyman's production - as Maple Leafs*

Assists - Hyman          |  Goals - Hyman
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Zach%20Hyman_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Zach%20Hyman_goal_success.png?raw=true)

## Still worried about Maple Leafs' top forward line post-Hyman

This leaves me with my annual disappointment with the Leafs. For the past couple seasons, Zach Hyman – Auston Matthews – Mitchel Marner was expected to be the most proficient forward line of the team. However, Leaf’s attempt to generate opportunities with the fast-pace line has been inconsistent at best. One of the reason may be that the team's top scorer, Matthews, may be attepting too much in wider region, as shown by the figure below,

Assists - Matthews          |  Goals - Matthews
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Auston%20Matthews_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Auston%20Matthews_goal_success.png?raw=true)

As a side note, it remains to be seen if Matthews’s overwhelming presence may have limited Hyman’s production. The limitation follows with Marner's, 

Assists - Marner          |  Goals - Marner
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Mitchell%20Marner_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Mitchell%20Marner_goal_success.png?raw=true)

Maple Leafs still have to answer if and how Matthews and Marner can provide the kitchen-sink approach that St. Louis leveraged (especially in the absence of speed-and-physicalness like Hyman’s). Perhaps, Leafs’ production is too depended on Matthews (only) and may require the team to strategize generating opportunities by other forwards.

## Methodology
I look at respective players’ shot attempts, assists on goals, and goals from 2016-2017 to 2020-2021 regular seasons. With the play-level data available with the [NHL API](https://thinkingjustin.com/collection/nhl_game_data_pull_eg.html), I aggregate the events mentioned based on the coordinates of the (half) rink. I collected all regular season events on offensive zone point-generating attempts (shots, blocked shots, missed shots (wide), and goals). The measures are then aggregated by players-of-interests mentioned in the text to calculate shooting percentages for each player. I repeate the exercise but with all skaters to get league's average. For each position on the rink, the relative difference between the player and the league average were compared to estimate over/under measurements presented in the plots. To provide a representative view, we only observe rink locations with shot attempts and goal scores more than 4 times each season.
 
[back to the main page](../index.md)
