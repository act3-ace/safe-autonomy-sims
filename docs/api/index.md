# API

Welcome to an overview of the safe-autonomy-sims repository's API. Navigate the hyperlinks on
the page or sidebar to the left to explore. The contents of this API page are generated based 
automatically from our code and docstrings defined within. 

The biggest organizational separation in our repository's file structure is represented below.
Modules either exist in the backend or the core. The backend encapsulates modules which define
entities and dynamics models required to accurately transition the state of a task's simulation. The
core encapsulates all CoRL extensions required to run our tasks using CoRL's framework. This includes
Agents, Dones, Glues, Platforms, Rewards, Run Time Assurance, and Simulators. 


- [backend](backend/index.md)
- [core](core/index.md)