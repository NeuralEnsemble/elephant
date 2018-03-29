def main(): # pragma: no cover
  from brian2 import start_scope,mvolt,ms,NeuronGroup,StateMonitor,run
  import matplotlib.pyplot as plt
  import neo
  import quantities as pq

  start_scope()
  
  # Izhikevich neuron parameters.  
  a = 0.02/ms
  b = 0.2/ms
  c = -65*mvolt
  d = 6*mvolt/ms
  I = 4*mvolt/ms
  
  # Standard Izhikevich neuron equations.  
  eqs = '''
  dv/dt = 0.04*v**2/(ms*mvolt) + (5/ms)*v + 140*mvolt/ms - u + I : volt
  du/dt = a*((b*v) - u) : volt/second
  '''
  
  reset = '''
  v = c
  u += d
  '''
  
  # Setup and run simulation.  
  G = NeuronGroup(1, eqs, threshold='v>30*mvolt', reset='v = -70*mvolt')
  G.v = -65*mvolt
  G.u = b*G.v
  M = StateMonitor(G, 'v', record=True)
  run(300*ms)
  
  # Store results in neo format.  
  vm = neo.core.AnalogSignal(M.v[0], units=pq.V, sampling_period=0.1*pq.ms)
  
  # Plot results.  
  plt.figure()
  plt.plot(vm.times*1000,vm*1000) # Plot mV and ms instead of V and s.  
  plt.xlabel('Time (ms)')
  plt.ylabel('mv')
  
  # Save results.  
  iom = neo.io.PyNNNumpyIO('spike_extraction_test_data')
  block = neo.core.Block()
  segment = neo.core.Segment()
  segment.analogsignals.append(vm)
  block.segments.append(segment)
  iom.write(block)
  
  # Load results.  
  iom2 = neo.io.PyNNNumpyIO('spike_extraction_test_data.npz')
  data = iom2.read()
  vm = data[0].segments[0].analogsignals[0]
  
  # Plot results. 
  # The two figures should match.   
  plt.figure()
  plt.plot(vm.times*1000,vm*1000) # Plot mV and ms instead of V and s.  
  plt.xlabel('Time (ms)')
  plt.ylabel('mv')
  
if __name__ == '__main__':
  main()
