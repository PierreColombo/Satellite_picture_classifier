import matplotlib.pyplot as plt

# computer crash if we try to run a cross validation program : we he to run one patch_size by one.
plt.plot([4,5,6,7,8] ,[1-0.7788,1-0.781871,1-0.7859,1-0.8016,1-0.82577], 'r--',[4,5,6,7,8] ,[1-0.7649,1-0.77558,1-0.7866,1-0.79076,1-0.76584], 'b--')
plt.title('Train Score & Validation Score, cross validation epoch')
plt.show()


