import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


plt.ion()
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
plt.show(block=False)
plt.pause(1)
import matplotlib
print(matplotlib.get_backend())
