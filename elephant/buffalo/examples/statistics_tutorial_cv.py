import matplotlib.pyplot as plt

from quantities import s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process

# Provenance tracking
from elephant.buffalo import decorator

decorator.activate()

spiketrain_list = [homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s,
                                               t_stop=100.0*s)
                   for i in range(100)]

plt.figure(dpi=150)
plt.eventplot(spiketrain_list, linelengths=0.75, linewidths=0.75,
              color="black")
plt.xlabel("Time, s")
plt.ylabel("Neuron id")
plt.xlim([0, 1])

from elephant.statistics import isi, cv
cv_list = [cv(isi(spiketrain)) for spiketrain in
           spiketrain_list]

# let's plot the histogram of CVs
plt.figure(dpi=100)
plt.hist(cv_list)
plt.xlabel("CV")
plt.ylabel("count")
plt.title("Coefficient of Variation of homogeneous Poisson process")

plt.show()

decorator.print_history()
decorator.save_graph("elephant_tutorial.html", show=True)
