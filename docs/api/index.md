# API

Welcome to an overview of the safe-autonomy-sims repository's API. Navigate the hyperlinks on
the page or sidebar to the left to explore. The contents of this API page are generated
automatically from our code and the docstrings defined within. 

The biggest organizational separation in our repository's file structure is represented below.
Modules either exist in the backend or the core. 

The backend contains custom aerospace simulators which define entities and dynamics models required to accurately transition the simulation state during a task episode.

The core encapsulates all CoRL extensions required to run our tasks using CoRL's framework. This includes
Agents, Dones, Glues, Platforms, Rewards, Run Time Assurance, and Simulators. 


- [backend](backend/index.md)
- [core](core/index.md)