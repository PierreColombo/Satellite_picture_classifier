import matplotlib.pyplot as plt

# computer crash if we try to run a cross validation program : we he to run one patch_size by one.
plt.plot([8,16,20,32,40] ,[1-0.768141,1-0.781871,1-0.78202,1-0.7880,1-0.7921], 'r--',[8,16,20,32,40] ,[1-0.7479229,1-0.77558,1-0.75653,1-0.7472,1-0.7459], 'b--')
plt.title('Train Score & Validation Score, cross validation patch size')
plt.show()
