# Offensive zone positional successes

## Highlights

*	Stanley Cup-winning teams have very specific strategies in controlling puck possessions and goal-scoring positions in offensive zones
* Edmonton’s Zach Hyman trade may complement where Connor McDavid had challenges
* Maple Leaf’s struggle, even with two of the top scorers in the league, may be depending too much on those star players

## What I aim to observe

Observing play-level statistics from 2016-17 seasons to 2020-21 seasons, I plot over (colored orange) and under (colored blue) shots-on-goal percentage of each players relative to the league average based on their position in offensive zone. See approach details on Methodology section. Along with the first-layer information (shots-on-goal statistics), the second layer (where the event happens) allows me to identify strategic positional patterns for each team and each player combinations. Unless otherwise mentioned in the plots, all figures observe player's shooting percentages (percentage of goals, of all shot attempts) - relative to the league average.

## Anecdotal successful player positions and strategies
I observe three recent Stanley Cup-winning teams: Washington Capitals (2018), St. Louis Blues (2019), and Tampa Bay Lightning (2020 and 2021). Especially, the top forward lines for the teams have distinct approaches that helped them raise the cup. Table below notes players for the top lines.

### Top Forward lines for each team

| Teams | Line Combinations |
| --- | :---: | 
| Washington Capitals | Ovechkin - Kuznetsov/Backstrom - Wilson/Oshi |
| St. Louis Blues | Schenn-Schwartz-Tarasenko |
| Tampa Bay Lightening | Point-Palat-Kucherov |

### Capital's stay-in-your-lane approach
Capitals benefit from a reliable goal-scoring from distance, that is from Alex Ovechkin. In order to synergize, other forwards of the line has been narrowly positioned near the goal crease. This effectively generates blocks and benefits secondary opportunities. 

*Washington Capital top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_caps_18.gif?raw=true)

### Kitchen-sink approach by St. Louis Blues
Blues in its 2018-2019 benefited from the top line's presence and scoring approach. With its scoring power, the line sinked its opposing goalies directly.

*St. Louis Blues top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_stl_2019.gif?raw=true)

### Kucherov we trust
Lightening’s all-in-one approach has a mixture of both strategies mentioned above. Centered on its top scorer, Nikita Kucherov, Lightening manages to position its forwards to open up opportunities and follow through shots missed or blocked.

*Tampa Bay Lightening top line - forwards*

![Alt Text](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/gif_tbl_2021.gif?raw=true)

## Oiler's Acquisition of Hyman may complement McDavid
I focus on my personal interest, Edmonton Oilers and Toronto Maple Leafs – in the context of Zach Hyman. I hypothesize that the acquisitino of Zach Hyman by Oilers will complement existing top scorers of the team (Connor McDavid and Leon Draisaitl) based on Hyman’s positional impact as Leafs forward.

Figure below (right) shows Connor McDavid’s goal scoring success on the rink. Surprisingly, McDavid’s success mainly comes away from the center.  However, his assist in goals comes closer to net (figure left). Meaning, McDavid’s production will be amplified (especially his production near goal crease if he is complemented in the region. An ideal candidate is a physical player who knows how to grind out the game and is not afraid of hard-skating or able to open up the offensive zone.

Assists - McDavid          |  Goals - McDavid
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Connor%20McDavid_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Connor%20McDavid_goal_success.png?raw=true)

While the gap is not apparent for Draisaitl,

Assists - Draisaitl          |  Goals - Draisaitl
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Leon%20Draisaitl_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Leon%20Draisaitl_goal_success.png?raw=true)

As a Leafs’ left winger, Hyman’s point production was not as stellar as Panarin or Gaudreau. However, his presence up front and better-than-average production right under goalie’s nose would complement where McDavid’s weaknesses would be. As Hyman is known to be a fast and hardest working skater , it would be exciting to watch the pair skate together.

*Hyman's production - as Maple Leafs*

Assists - Hyman          |  Goals - Hyman
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Zach%20Hyman_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Zach%20Hyman_goal_success.png?raw=true)

## Still worried about Maple Leafs’s top forward line post-Hyman

This leaves me with my annual disappointment with the Leafs. Up until the last season, the forward line, Zach Hyman – Auston Matthews – Mitchel Marner, has been the most proficient forward line of the team. Complementary of Taveras in other lines, however, Leaf’s attempt of fast-pace team still yielded to inconsistent results.

Assists - Matthews          |  Goals - Matthews
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Auston%20Matthews_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Auston%20Matthews_goal_success.png?raw=true)

As a side note, it remains to be seen if Matthews’s overwhelming presence may have limited Hyman’s production. 

Assists - Marner          |  Goals - Marner
:-------------------------:|:-------------------------:
![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Mitchell%20Marner_assist_success.png?raw=true)  |  ![](https://github.com/justinjoliver/NHL-Analytics/blob/main/dev/positional_success_forwards/Mitchell%20Marner_goal_success.png?raw=true)

Maple Leafs still have to answer if the top two players, Matthews and Marner, can provide the kitchen-sink approach that St. Louis leveraged (especially in the absence of speed-and-physicalness like Hyman’s). Perhaps, Leafs’ production is too depended on Matthews, and Matthews only, at this point.

## Methodology
I look at respective players’ shot attempts, assists on goals, and goals from 2016-2017 to 2020-2021 regular seasons. With the play-level data available with the [API pull](https://thinkingjustin.com/collection/nhl_game_data_pull_eg.html), I aggregate the events mentioned above based on the coordinates of the (half) rink. As for the measurements, I collected all regular season events relevant to offensive zone point-generating attempts (shots, blocked shots, missed shots (wide), and goals). The measures are then aggregated by players-of-interests mentioned above to calculate shooting percentage for each player. I repeate the exercise but with all skaters to get league's average. For each position on the rink, the relative difference between the player and the league average were compared to estimate over/under measurements mentioned in the main text. To provide a representative view, we only observe rink locations with shot attempts and goal scores more than 4 times each season.
 
[back to the main page](../index.md)
