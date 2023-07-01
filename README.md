# Hill Climbing


![hill_climbing](https://github.com/timothy/hill_climbing/assets/7410132/697e641c-99fe-46ac-afc3-f9eb18ca9503)

## pseudo code for local maximum
```python
function HILL-CLIMBING(problem) # returns the a local maximum
	current <- problem.INITIAL
	while true do
		neighbor <- nextHighest # Look left, look right, choose the higher value
		if VALUE(neighbor) <= VALUE(current): return current # reached local peak
		current <- neighbor
		
```
