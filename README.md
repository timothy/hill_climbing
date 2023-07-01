# Hill Climbing


![hill_climbing](https://github.com/timothy/hill_climbing/assets/7410132/697e641c-99fe-46ac-afc3-f9eb18ca9503)

Hill climbing can get stuck for the below reasons
 - Local maxima
 - Ridges: Result in a sequence of local maxima that is very difficult fro greedy algorithms to navigate
 - Plateaus: flat spaces like the sholder

## pseudo code for local maximum hill climb
```python
function HILL-CLIMBING(problem) # returns the a local maximum
	current <- problem.INITIAL
	while true do
		neighbor <- nextHighest # Look left, look right, choose the higher value
		if VALUE(neighbor) <= VALUE(current): return current # reached local peak
		current <- neighbor
		
```


### Algorithms
 - Stochastic hill climbing
 - First-choice hill climbing: implements stochastic hill climbing by generating successors randomly
 - Random-restart hill climbing (if you don't succeed try try again)
