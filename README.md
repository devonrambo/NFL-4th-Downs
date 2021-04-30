# NFL 4th Down Analysis

### Description
This project uses a Kaggle database (www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016) containing every regular season play from the 2009 – 2016 NFL seasons. This provided over 30,000 4th down plays examine. Multiple decision trees and random forests were built to examine the factors that go into coaches’ 4th down decisions. Situational outcomes were also examined using expected points added (EPA).

### Motivation
A common idea on “Analytics Twitter” is that NFL teams leave value on the table by refusing to go for it on 4th down. Football is referred to as “A Game of Inches” and we’re reaching a point where every edge a team can get, not just schematically but also strategically, is being explored. I used decision trees and random forests to zoom in on some of the causes and effects of 4th down decisions. 

### Methods

##### Decision Factor Analysis
⦁	A decision tree algorithm was used to predict whether a given fourth down would result in a punt, field goal, or going for it with 89.9% accuracy. <br />
⦁	The factors the algorithm used were time left in the game, field position, yards to the first down, whether it was a goal-to-go situation, and score differential. <br />
⦁	A random forest algorithm was then implemented and increased the accuracy up to 92.6%. <br />
⦁	The weight of the deciding factors from the algorithm were then plotted to see what coaches generally base their decisions on. <br /> <br />



##### Decision Results Analysis

⦁	After building out the model that can accurately predict coaches’ behavior, it was time to examine the value lost/gained by these decisions. <br />
⦁	Decisions made in various areas of the field (including what I’ve labeled the “Deadzone”) were examined and the average EPA was calculated for different situations.  <br />
⦁	The Deadzone is from the 50 yard line to the opponents 35 yard line, where field goals are inefficient, and coaches often punt rather than going-for-it. <br />
⦁	After identifying the situations with the most lost value, another random forest algorithm was used to check out the deciding factors and shed light on the traps coaches fall into. <br />


### Visuals

Check out some of the charts / graphs generated from the model in the files above!

### Sources

The data was pulled from Kaggle user Max Horowitz's database. The link is in the description!
