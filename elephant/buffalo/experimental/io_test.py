import quantities as pq
import matplotlib.pyplot as plt
from elephant.buffalo.examples.run_isi_histogram import load_data
#import elephant.buffalo.experimental.io as io_monitor


def do_plot(block, show=False):
    figure = plt.figure()
    plt.eventplot(block.segments[0].spiketrains[0].times.rescale(pq.s))
    plt.xlim(0, 60)
    if show:
        plt.show()
    return figure


if __name__ == "__main__":
 #   io_monitor.activate()
    input_files = set()
    output_files = set()
  #  io_monitor.monitor.set_callbacks(input_files, output_files)

    #plt.Figure.savefig = io_monitor.savefig_callback(plt.Figure.savefig)

    block = load_data("i140703-001", [10])

    figure = do_plot(block)
    plt.figure(figure.number)
#    figure.savefig("io_test.png")
    plt.savefig("io_test2.png")

   # io_monitor.deactivate()

    print("Inputs: {}".format(input_files))
    print("Outputs: {}".format(output_files))
