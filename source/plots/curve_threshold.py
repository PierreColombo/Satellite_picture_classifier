import matplotlib.pyplot as plt
import numpy as np

# computer crash if we try to run a cross validation program : we he to run one patch_size by one.
plt.plot([0.3,0.4,0.5,0.6,0.7,0.75] ,[1-0.857050875029,1-0.867875575246,1-0.872438214454,1-0.87354756428,1-0.872908150873,1-0.871656932239], 'r--')
plt.title('Test error on an sample of 40 pictures with different thresholds')
plt.show()

