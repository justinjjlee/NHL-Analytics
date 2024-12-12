<h1>
<a href="https://medium.com/@thinkingjustin">
<img src="docs/images/logo_bing.jpeg" width="80px" align="left" style="margin-right: 10px;", alt="nhla-logo"> 
</a> NHL Analytics: A Ice Hockey Sports Analytics Platform Based on National Hockey League (NHL) Data
</h1>

If you find my work to be useful, please star this repository!

[![justinjjlee - NHL-Analytics](https://img.shields.io/static/v1?label=justinjjlee&message=NHL-Analytics&color=blue&logo=github)](https://github.com/justinjjlee/NHL-Analytics "Go to GitHub repo")
[![stars - NHL-Analytics](https://img.shields.io/github/stars/justinjjlee/NHL-Analytics?style=social)](https://github.com/justinjjlee/NHL-Analytics)
[![forks - NHL-Analytics](https://img.shields.io/github/forks/justinjjlee/NHL-Analytics?style=social)](https://github.com/justinjjlee/NHL-Analytics)
[![Medium - NHL-Analytics](https://img.shields.io/badge/Medium-Read-green?logo=Medium)](https://medium.com/@thinkingjustin)
[![Streamlit - NHL-Analytics](https://img.shields.io/badge/Streamlit-Explore-FF4B4B?style=flat&logo=streamlit&logoColor=white/)](https://share.streamlit.io/user/justinjjlee)

This is a collection of methods for collecting, compiling, cleaning, analyzing, modeling, and predicting team and player (skaters and goalies) performances and strategies. This repository does not claim ownership of the data and reflects the perspectives of the organizations or entities mentioned. All original code (including generic and model algorithms) may be used freely, provided proper citation and credit are given to this repository.

## Analysis & Insights
All of my analyses and deep-dive insights are written and presented in **[my Medium blog](https://medium.com/@thinkingjustin)**.

## Goals & Capabilities
I use publicly available data to build up the analytics capabilities and insights generated beyond the headline statics easily measurable. 

 * Capture complex strategical, behavioral, and performance trends asked by fans of the sport
 * Integrate different data sources (e.g. college hockey roaster and building up performance trend beyond players' professional career)

I hope works saved in this repository allows for replications, explorations, and advancing new measurements and insights.

<details>
<summary><strong><em>Applied Tools</em></strong></summary>

Capabilities I use for data collection, processing, and analysis to derive insights, data visualizations, and predictive models.

| Capability | Tools used |
| --- | --- |
| General |  <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/python.svg" width="32"/> <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/julia.svg" width="32"/> |
| Data Collection & Processing | <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/duckdb.svg" width="32"/> |
| ML Model Build | <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/pytorch.svg" width="32"/> <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/scikitlearn.svg" width="32"/>|
| Interactive Data Visualization | <img src="https://github.com/simple-icons/simple-icons/blob/develop/icons/tableau.svg" width="32"/> |

</details>

<details>
<summary><strong><em>Data Pull & Process Automation with Github Actions</em></strong></summary>

The **Github Actions** is being used to update the data saved in this repository folder `./latest/`. The data collection is run every day.

 * Team-level rank
 * Game-level stats
 * Game-level betting odds
 * Play-by-play records

Required package version used is saved in `./src/requirement` through `.sh` command. Note that the python environment function pull is based on where the script is located, where as data file reference is based on Github repository head directory.

</details>

```
   ,
    -   \O                                     ,  .-.___
  -     /\                                   O/  /xx\XXX\
 -   __/\ `\                                 /\  |xx|XXX|
    `    \, \_ =                          _/` << |xx|XXX|
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
```
