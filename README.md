# SBM_Simulation
Main repository for epidemiological MC-simulations on stochastic blockmodels

## 

Offloads $N_{graph}\times N_{sim}$ simulations to a given SYCL-device, 
where vertices of graphs are grouped into `communities`, and edges between communities are grouped into `connections`. 

Each pack of $N_{sim}$ simulations are allocated as an individual unit, and its kernels are enqueued to a single (shared) SYCL queue separately.

## Buffer Allocations

### Population States
Each simulation allocates $N_{pop}\times N_{partition}$ vertex states ($S$, $I$, or $R$) at each timestep $t$ for vertices.
$N_{partition}$ SIR-model states are allocated for each timestep $t$, in order to accumulate SIR population counts.

### Tracking Infections
A buffer with $N_{connection}$ integers at each timestep $t$ is allocated to keep track of the total number of infection events occuring
between communities at each timestep.

### Graph
All graph properties are allocated to global buffers shared between the $N_{sim}$ simulations.
$N_{edge}$ edges with (`uint32_t` from, `uint32_t` to) indices are allocated
An `edge_connection_map` (with `uint32_t` `connection`) is allocated to determine which `connection` each edge belongs to.
A `vertex_partition_map` (with `uint32_t` `partition`) is allocated to determine which `partition` each vertex belongs to.


## Memory Synchronization
When simulations surpass a given number of timesteps $N_{t,alloc}$, results are transferred from the compute device to the host.

## MC Simulation Workflow
![]relative%20Plot/SBM_Simulation_Workflow.png?raw=true)


## Top-level

A `connection_community_map` (with `uint32_t` from, `uint32_t` to) is used to determine which `community` each `connection` belongs to.
