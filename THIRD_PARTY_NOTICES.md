# Source and component notices

When relevant, please cite [A Safe Action Is Not Enough: Feasible-Future
Decoding for Vision-Language-Action Policies](https://arxiv.org/abs/2610.05166)
by Nguyen et al.

This project contains generic rollout and search APIs, an independent
benchmark adapter, and simulator observation/cost helpers. The adapter calls
external Safety-CHORES task samplers and accepts an independent policy interface.
No model checkpoint is included or required by its uniform-random baseline.

Safety-CHORES tasks, the SafeVLA benchmark implementation, AI2-THOR, Unity,
ProcTHOR/Objaverse assets, and Python dependencies retain their own provenance and
license terms. These external source trees, binaries, task data, and assets are
not bundled in this repository. Runtime configuration pins local external sources and
artifacts by SHA256.

The observation/cost helpers were developed for the modified simulator and
retained separately from controller implementations. This repository does not grant
an open-source license.
